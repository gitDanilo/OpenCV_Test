#include <iostream>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/calib3d/calib3d.hpp>

using namespace std;
using namespace cv;

#define MAX_DISTANCE 9999.0

#define MIN_HAND_DEFECTS 4 // min 4 hand defects, one between each finger
#define MAX_HAND_DEFECTS 5 // max number of palm points
#define DELTA_LENGTH 0.7f

// Hand defect thresholds
#define MIN_FINGER_WIDTH 25
#define MAX_FINGER_WIDTH 60
#define MIN_FINGER_LENGTH 60
#define MAX_FINGER_LENGTH 220
#define MIN_INNER_ANGLE 15 // min angle formed by the hand defect
#define MAX_INNER_ANGLE 115 // max angle formed by the hand defect

// Palm thresholds
#define MIN_PALM_LENGTH 10
#define MAX_PALM_LENGTH 220
#define MIN_PALM_INNER_ANGLE 10
#define MAX_PALM_INNER_ANGLE 179

typedef struct _HandDefect
{
	// startPoint, endPoint, farthestPoint points returned by convexityDefects
	Point2f startPoint;
	Point2f endPoint;
	Point2f farthestPoint;
	float length;
} HandDefect, *PHandDefect;

static const Scalar mMinHSV(0, 30, 60), mMaxHSV(20, 150, 255); // HSV of avarage human skin tone
static const Size mSize(624, 832); // size of the displayed image
// Method 1 of detecting palm center
static Point2f mPalmCenter;
static float mPalmRadius;
// Method 2 of detecting palm center
static Point2f mPalmCenter2;
static float mPalmRadius2;

// Returns the distance of 2 points
inline float distanceP2P(Point2f a, Point2f b)
{
	return sqrt(pow(a.x - b.x, 2) + pow(a.y - b.y, 2));
}

// Returns the closest 3 points that respect the min and max threshold distances
int find3ClosestPoints(const vector<Point2f> &points, float *d)
{
	if (points.size() < MIN_HAND_DEFECTS && d == NULL)
		return -1;

	d[0] = d[1] = d[2] = 0;

	int i;
	vector<Point2f>::size_type size;
	i = size = points.size();

	float shortestDist = MAX_DISTANCE;
	float oldDist;
	int result = -1;

	for (; i > 0; --i)
	{
		oldDist = d[0];
		d[0] = distanceP2P(points[i % size], points[(i - 1) % size]);
		if (d[0] < MIN_FINGER_WIDTH || d[0] > MAX_FINGER_WIDTH)
		{
			// skip next point because he will be invalid
			d[0] = 0;
			--i;
			continue;
		}
		d[1] = oldDist > 0 ? oldDist : distanceP2P(points[i % size], points[(i + 1) % size]);
		if (d[1] < MIN_FINGER_WIDTH || d[1] > MAX_FINGER_WIDTH)
			continue;

		d[2] = d[0] + d[1];

		if ((d[2]) < shortestDist)
		{
			shortestDist = d[2];
			result = i % size;
		}
	}

	return result;
}

// Returns the closest 3 points that respect the min and max threshold distances
int find3ClosestPoints(const PHandDefect defects, int defectsSize, float *d)
{
	if (defectsSize < MIN_HAND_DEFECTS && d == NULL)
		return -1;

	d[0] = d[1] = d[2] = 0;

	int i = defectsSize;
	float shortestDist = MAX_DISTANCE;
	float oldDist;
	int result = -1;

	for (; i > 0; --i)
	{
		oldDist = d[0];
		d[0] = distanceP2P(defects[i % defectsSize].farthestPoint, defects[(i - 1) % defectsSize].farthestPoint);
		if (d[0] < MIN_FINGER_WIDTH || d[0] > MAX_FINGER_WIDTH)
		{
			// skip next point because he will be invalid
			d[0] = 0;
			--i;
			continue;
		}
		d[1] = oldDist > 0 ? oldDist : distanceP2P(defects[i % defectsSize].farthestPoint, defects[(i + 1) % defectsSize].farthestPoint);
		if (d[1] < MIN_FINGER_WIDTH || d[1] > MAX_FINGER_WIDTH)
			continue;

		d[2] = d[0] + d[1];

		if ((d[2]) < shortestDist)
		{
			shortestDist = d[2];
			result = i % defectsSize;
		}
	}

	return result;
}

// Returns the C angle on a triangle (A, B, C)
float innerAngle(Point2f a, Point2f b, Point2f c)
{
	float CAx = c.x - a.x;
	float CAy = c.y - a.y;
	float CBx = c.x - b.x;
	float CBy = c.y - b.y;

	// https://www.mathsisfun.com/algebra/trig-cosine-law.html (The Law of Cosines)
	float A = acos((CBx*CAx + CBy*CAy) / (sqrt(CBx*CBx + CBy*CBy) * sqrt(CAx*CAx + CAy*CAy)));	// (a² + b² − c²) / 2
																								// ( (sqrt( (Ax - Bx)*(Ax - Bx) + (Ay - By)*(Ay - By) ))² + (sqrt( (Ax - Cx)*(Ax - Cx) + (Ay - Cy)*(Ay - Cy) ))² − (sqrt( (Bx - Cx)*(Bx - Cx) + (By - Cy)*(By - Cy) ))²) / 2

	return A * 180.0f / CV_PI;
}

// Returns arcTang of 2 points in OpenCV space (inverted Y)
inline float arcTang(Point2f a, Point2f b, bool rad)
{
	return rad ? atan2(a.y - b.y, b.x - a.x) : atan2(a.y - b.y, b.x - a.x) * 180.0f / CV_PI;
}

void draw(Mat &frame, const PHandDefect defects, int defectsSize)
{
	circle(frame, mPalmCenter, mPalmRadius, Scalar(255, 0, 0), 1);
	circle(frame, mPalmCenter, 4, Scalar(255, 0, 0), 2);
	for (int i = 0; i < defectsSize; ++i)
	{
		circle(frame, defects[i].startPoint, 4, Scalar(255, 0, 0), 2);
		circle(frame, defects[i].endPoint, 4, Scalar(0, 255, 0), 2);
		circle(frame, defects[i].farthestPoint, 4, Scalar(0, 0, 255), 2);
		line(frame, defects[i].startPoint, defects[i].endPoint, Scalar(255, 0, 255), 1);
		line(frame, defects[i].farthestPoint, defects[i].startPoint, Scalar(255, 0, 255), 1);
		line(frame, defects[i].farthestPoint, defects[i].endPoint, Scalar(255, 0, 255), 1);
	}
}

// Displays the arcTang of mPalmCenter and the cursor position on the image
void CallbackFunc(int event, int x, int y, int flags, void* userdata)
{
	if (event == cv::EVENT_LBUTTONUP)
	{
		float angle = arcTang(mPalmCenter, Point2f(x, y), false);
		cout << "angle: " << angle << endl;
	}
}

int main()
{
	const char* windowName = "OpenCV_Test";
	namedWindow(windowName);
	setMouseCallback(windowName, CallbackFunc, NULL);
	//VideoCapture cap(0);

	Mat img;
	Mat bgr;
	Mat hsv;

	HandDefect listDefects[MAX_HAND_DEFECTS] = {0};
	int listDefectsSize = 0;

	do
	{
		//cap >> frame;
		img = imread("C:\\Users\\danil\\source\\repos\\OpenCV_Test\\Debug\\myhand13.jpg", 1);
		resize(img, bgr, mSize);

		cvtColor(bgr, hsv, CV_BGR2HSV);
		inRange(hsv, mMinHSV, mMaxHSV, hsv);

		// Pre processing
		//int elementSize = 3;
		medianBlur(hsv, hsv, 5);
		//Mat element = getStructuringElement(MORPH_ELLIPSE, Size(2 * elementSize + 1, 2 * elementSize + 1), Point(elementSize, elementSize));
		//dilate(hsv, hsv, element);

		// Contour detection
		vector<vector<Point>> contours;
		vector<Vec4i> hierarchy;
		findContours(hsv, contours, hierarchy, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_SIMPLE, Point(0, 0));

		if (!contours.empty())
		{
			size_t largestContour = 0;
			for (size_t i = 1; i < contours.size(); i++)
			{
				if (contourArea(contours[i]) > contourArea(contours[largestContour]))
					largestContour = i;
			}
			drawContours(bgr, contours, largestContour, Scalar(0, 255, 0), 1);

			vector<int> hullIndexes;
			convexHull(Mat(contours[largestContour]), hullIndexes, false);

			//vector<Point> aa;
			//convexHull(Mat(contours[largestContour]), aa, false);
			//approxPolyDP(aa, aa, 24, true);
			//polylines(bgr, aa, true, Scalar(255, 0, 0), 1);

			if (!hullIndexes.empty())
			{
				vector<Vec4i> defects;
				convexityDefects(Mat(contours[largestContour]), hullIndexes, defects);

				listDefectsSize = defects.size();
				if (listDefectsSize >= MIN_HAND_DEFECTS)
				{
					float angle;
					float length;
					int i = 0, j = 0;

					Point2f averagePoint;
					vector<Point2f> palmPoints;
					palmPoints.reserve(MAX_HAND_DEFECTS);


					// The convexity defects which have depth larger than a threshold value tend to appear around the palm portion
					for (; i < listDefectsSize; ++i)
					{
						length = defects[i][3] / 256.0f;
						angle = innerAngle(contours[largestContour][defects[i][0]],
										   contours[largestContour][defects[i][1]],
										   contours[largestContour][defects[i][2]]);

						// Hand defect thresholds
						if (length >= MIN_FINGER_LENGTH &&
							length <= MAX_FINGER_LENGTH &&
							angle >= MIN_INNER_ANGLE &&
							angle <= MAX_INNER_ANGLE)
						{
							if (j < MAX_HAND_DEFECTS)
							{
								listDefects[j].startPoint = contours[largestContour][defects[i][0]];
								listDefects[j].endPoint = contours[largestContour][defects[i][1]];
								listDefects[j].farthestPoint = contours[largestContour][defects[i][2]];
								listDefects[j].length = length;
								++j;
							}
							else
							{
								j = 0;
								break;
							}
						}

						// Palm thresholds
						if (length >= MIN_PALM_LENGTH &&
							length <= MAX_PALM_LENGTH &&
							angle >= MIN_PALM_INNER_ANGLE &&
							angle <= MAX_PALM_INNER_ANGLE)
						{
							palmPoints.push_back(contours[largestContour][defects[i][2]]);
							averagePoint.x += contours[largestContour][defects[i][2]].x;
							averagePoint.y += contours[largestContour][defects[i][2]].y;
						}
					}

					if (j >= MIN_HAND_DEFECTS && j <= MAX_HAND_DEFECTS)
					{
						minEnclosingCircle(palmPoints, mPalmCenter2, mPalmRadius2);

						// Combine the average position of all the depth points with the min enclosing circle center
						averagePoint.x += mPalmCenter2.x;
						averagePoint.y += mPalmCenter2.y;
						averagePoint.x /= palmPoints.size() + 1;
						averagePoint.y /= palmPoints.size() + 1;

						float d[3];
						//int index = find3ClosestPoints(palmPoints, d);
						int index = find3ClosestPoints(listDefects, j, d);
						if (index != -1)
						{
							Point2f palmPoint = listDefects[(index + 1) % j].farthestPoint;
							Point2f midPoint;

							if (index == 0)
							{
								midPoint.x = (listDefects[j - 1].farthestPoint.x + listDefects[(index + 1) % j].farthestPoint.x) / 2.0f;
								midPoint.y = (listDefects[j - 1].farthestPoint.y + listDefects[(index + 1) % j].farthestPoint.y) / 2.0f;
							}
							else
							{
								midPoint.x = (listDefects[index - 1].farthestPoint.x + listDefects[(index + 1) % j].farthestPoint.x) / 2.0f;
								midPoint.y = (listDefects[index - 1].farthestPoint.y + listDefects[(index + 1) % j].farthestPoint.y) / 2.0f;
							}

							//Point2f midPoint((listDefects[index - 1].farthestPoint.x + listDefects[index].farthestPoint.x + listDefects[index + 1].farthestPoint.x) / 3.0f,
							//				 (listDefects[index - 1].farthestPoint.y + listDefects[index].farthestPoint.y + listDefects[index + 1].farthestPoint.y) / 3.0f);

							angle = arcTang(midPoint, palmPoint, true);
							mPalmRadius = listDefects[index].length * DELTA_LENGTH;

							// Rotate mPalmCenter point by angle
							mPalmCenter.x = -sin(angle) * mPalmRadius;
							mPalmCenter.y = cos(angle) * mPalmRadius;

							// Translate mPalmCenter to midPoint
							mPalmCenter.x += midPoint.x;
							mPalmCenter.y -= midPoint.y; // Inverted y on OpenCV
							//mPalmCenter.y = mPalmCenter.y - midPoint.y;
							mPalmCenter.y *= -1; // Inverted y on OpenCV

							// Draw the results
							//circle(bgr, midPoint, 4, Scalar(0, 0, 255), 2);
							//circle(bgr, palmPoint, 4, Scalar(0, 255, 0), 2);
							//circle(bgr, mPalmCenter, 4, Scalar(255, 0, 0), 2);
							//line(bgr, midPoint, palmPoint, Scalar(0, 255, 0), 1);
							//line(bgr, midPoint, mPalmCenter, Scalar(0, 255, 0), 1);
							//line(bgr, palmPoint, mPalmCenter, Scalar(0, 255, 0), 1);
							draw(bgr, listDefects, j);
						}
					}
				}
			}
		}

		imshow(windowName, bgr);
	} while (waitKey(30) < 0);

	return 0;
}
