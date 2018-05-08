#include <iostream>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/calib3d/calib3d.hpp>

using namespace std;
using namespace cv;

#define MAX_DISTANCE 9999.0
#define MAX_PALM_POINTS 7
#define MIN_DIST_THRESHOLD 25
#define MAX_DIST_THRESHOLD 60
#define MIN_HAND_DEFECTS 4
#define MAX_HAND_DEFECTS 5
#define MIN_LENGTH_VARIATION 20
#define MAX_LENGTH_VARIATION 240
#define MIN_INNER_ANGLE 15
#define MAX_INNER_ANGLE 115
#define MIN_PALM_INNER_ANGLE 10
#define MAX_PALM_INNER_ANGLE 175

typedef struct _HandDefect
{
	// startPoint, endPoint, farthestPoint points returned by convexityDefects
	Point2d startPoint;
	Point2d endPoint;
	Point2d farthestPoint;
	double innerAngle; // angle of the farthestPoint in a triangle (startPoint, endPoint, farthestPoint)
	// centAngle[0] represents the atang angle of the mCenter point with startPoint
	// centAngle[1] represents the atang angle of the mCenter point with endPoint
	double centAngle[2];
	// unitCircle[0] represents the unit circle of the startPoint
	// unitCircle[0] represents the unit circle of the endPoint
	int unitCircle[2];
	double length; // distance of startPoint to farthestPoint
} HandDefect, *PHandDefect;

//static const int mUnitCircleIndexes[4][2] = {{0, 1}, {0, 2}, {3, 1}, {3, 2}};
static const Scalar mMinHSV(0, 30, 60), mMaxHSV(20, 150, 255); // HSV of avarage human skin tone
static const Size mSize(624, 832); // size of the displayed image
static Point2f mCenter; // center of the unit circle

// Returns the distance of 2 points
inline double distanceP2P(Point2d a, Point2d b)
{
	return sqrt(pow(a.x - b.x, 2) + pow(a.y - b.y, 2));
}

// Returns the unit circle of an angle
// 0 = 0°..90°
// 1 = 90°..180°
// 2 = -180°.. - 90°
// 3 = -90°..0°
inline int getUnitCircle(double a)
{
	if (a > 0.0 && a <= 90.0)
		return 0;
	if (a > 90.0 && a <= 180.0)
		return 1;
	if (a > -180.0 && a <= -90.0)
		return 3;
	//if (a > -90.0 && a <= 0)
	return 2;
}

int find3ClosestPoints(const vector<Point2f> &points, double *d)
{
	if (points.size() < 4 && d == NULL)
		return -1;

	d[0] = d[1] = d[2] = 0;

	int i;
	vector<Point2f>::size_type size;
	i = size = points.size();

	double shortestDist = MAX_DISTANCE;
	double oldDist;
	int result = -1;

	for (; i > 0; --i)
	{
		oldDist = d[0];
		d[0] = distanceP2P(points[i % size], points[(i - 1) % size]);
		if (d[0] < MIN_DIST_THRESHOLD || d[0] > MAX_DIST_THRESHOLD)
		{
			// skip next point because he will be invalid
			d[0] = 0;
			--i;
			continue;
		}
		d[1] = oldDist > 0 ? oldDist : distanceP2P(points[i % size], points[(i + 1) % size]);
		if (d[1] < MIN_DIST_THRESHOLD || d[1] > MAX_DIST_THRESHOLD)
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

// Returns the C angle on a triangle (A, B, C)
double innerAngle(Point2d a, Point2d b, Point2d c)
{
	double CAx = c.x - a.x;
	double CAy = c.y - a.y;
	double CBx = c.x - b.x;
	double CBy = c.y - b.y;

	// https://www.mathsisfun.com/algebra/trig-cosine-law.html (The Law of Cosines)
	double A = acos((CBx*CAx + CBy*CAy) / (sqrt(CBx*CBx + CBy*CBy) * sqrt(CAx*CAx + CAy*CAy))); // (a² + b² − c²) / 2
																								// ( (sqrt( (Ax - Bx)*(Ax - Bx) + (Ay - By)*(Ay - By) ))² + (sqrt( (Ax - Cx)*(Ax - Cx) + (Ay - Cy)*(Ay - Cy) ))² − (sqrt( (Bx - Cx)*(Bx - Cx) + (By - Cy)*(By - Cy) ))²) / 2

	return A * 180 / CV_PI;
}

// Returns arcTang of 2 points
inline double arcTang(Point2d a, Point2d b)
{
	return atan2(a.y - b.y, a.x - b.x) * 180 / CV_PI;
}

// Displays the arcTang of mCenter and the cursor position on the image
void CallbackFunc(int event, int x, int y, int flags, void* userdata)
{
	if (event == cv::EVENT_LBUTTONUP)
	{
		double angle = atan2(mCenter.y - y, mCenter.x - x) * 180 / CV_PI;
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
		img = imread("C:\\Users\\danil\\source\\repos\\OpenCV_Test\\Debug\\myhand.jpg", 1);
		resize(img, bgr, mSize);

		cvtColor(bgr, hsv, CV_BGR2HSV);
		inRange(hsv, mMinHSV, mMaxHSV, hsv);

		// Pre processing
		medianBlur(hsv, hsv, 5);
		//Mat element = getStructuringElement(MORPH_ELLIPSE, Size(2 * elementSize + 1, 2 * elementSize + 1), Point(elementSize, elementSize));
		//dilate(hsv, hsv, element);
		//morphOps(hsv);

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
			drawContours(bgr, contours, largestContour, Scalar(0, 0, 255), 1);

			vector<int> hullIndexes;
			convexHull(Mat(contours[largestContour]), hullIndexes, false);

			//approxPolyDP(Mat(hull[0]), hull[0], 18, true);
			//drawContours(mFrame, hull, 0, Scalar(0, 255, 255), 2);

			if (!hullIndexes.empty())
			{
				vector<Vec4i> defects;
				convexityDefects(Mat(contours[largestContour]), hullIndexes, defects);

				listDefectsSize = defects.size();
				if (listDefectsSize >= MIN_HAND_DEFECTS)
				{
					double angle;
					double length;
					float radius;
					int i = 0, j = 0;

					vector<Point2f> palmPoints;
					Point2f averagePoint;
					palmPoints.reserve(MAX_PALM_POINTS);

					int fingerIndex = 0;
					char sText[16] = {0};

					// The convexity defects which have depth larger than a threshold value tend to appear around the palm portion
					for (; i < listDefectsSize; ++i)
					{
						length = defects[i][3] / 256.0;
						angle = innerAngle(contours[largestContour][defects[i][0]], contours[largestContour][defects[i][1]], contours[largestContour][defects[i][2]]);
						if (length >= MIN_LENGTH_VARIATION &&
							length <= MAX_LENGTH_VARIATION //&&
							//angle >= MIN_PALM_INNER_ANGLE &&
							/*angle <= MAX_PALM_INNER_ANGLE*/)
						{
							circle(bgr, contours[largestContour][defects[i][2]], 4, Scalar(0, 0, 255), 2);
							_itoa(fingerIndex++, sText, 10);
							putText(bgr, sText, contours[largestContour][defects[i][2]], CV_FONT_NORMAL, 0.7, Scalar(255, 255, 255));

							palmPoints.push_back(contours[largestContour][defects[i][2]]); // check for MAX_PALM_POINTS
							averagePoint.x += contours[largestContour][defects[i][2]].x;
							averagePoint.y += contours[largestContour][defects[i][2]].y;
						}
					}

					averagePoint.x /= palmPoints.size();
					averagePoint.y /= palmPoints.size();
					circle(bgr, averagePoint, 4, Scalar(255, 255, 0), 2);

					minEnclosingCircle(palmPoints, mCenter, radius);

					averagePoint.x = (averagePoint.x + mCenter.x) / 2.0;
					averagePoint.y = (averagePoint.y + mCenter.y) / 2.0;

					circle(bgr, mCenter, radius, Scalar(255, 0, 0));
					circle(bgr, mCenter, 4, Scalar(0, 255, 255), 2);
					circle(bgr, averagePoint, 4, Scalar(255, 255, 255), 2);

					double d[3];
					int index = find3ClosestPoints(palmPoints, d);
					if (index != -1)
					{
						Point2f midPoint;
						midPoint.x = (palmPoints[index - 1].x + palmPoints[index + 1].x) / 2.0;
						midPoint.y = (palmPoints[index - 1].y + palmPoints[index + 1].y) / 2.0;
						circle(bgr, midPoint, 4, Scalar(0, 255, 0), 2);

					}



					for (i = 0; i < listDefectsSize; ++i)
					{
						angle = innerAngle(contours[largestContour][defects[i][0]], contours[largestContour][defects[i][1]], contours[largestContour][defects[i][2]]);
						if (angle >= MIN_INNER_ANGLE &&
							angle <= MAX_INNER_ANGLE)
						{
							listDefects[j].startPoint = contours[largestContour][defects[i][0]];
							listDefects[j].endPoint = contours[largestContour][defects[i][1]];
							listDefects[j].farthestPoint = contours[largestContour][defects[i][2]];
							listDefects[j].innerAngle = angle;
							listDefects[j].centAngle[0] = arcTang(mCenter, listDefects[j].startPoint);
							listDefects[j].centAngle[1] = arcTang(mCenter, listDefects[j].endPoint);
							listDefects[j].unitCircle[0] = getUnitCircle(listDefects[j].centAngle[0]);
							listDefects[j].unitCircle[1] = getUnitCircle(listDefects[j].centAngle[1]);
							listDefects[j].length = distanceP2P(listDefects[j].startPoint, listDefects[j].farthestPoint);

							circle(bgr, listDefects[j].startPoint, 4, Scalar(255, 0, 0), 2);
							circle(bgr, listDefects[j].endPoint, 4, Scalar(0, 255, 0), 2);
							circle(bgr, listDefects[j].farthestPoint, 4, Scalar(0, 0, 255), 2);
							line(bgr, listDefects[j].startPoint, listDefects[j].endPoint, Scalar(0, 255, 0), 1);
							line(bgr, listDefects[j].startPoint, listDefects[j].farthestPoint, Scalar(0, 255, 0), 1);
							line(bgr, listDefects[j].endPoint, listDefects[j].farthestPoint, Scalar(0, 255, 0), 1);

							++j;
						}
					}

					//if (j >= MIN_HAND_DEFECTS && j <= MAX_HAND_DEFECTS)
					//{
					//}

				}
			}
		}

		imshow(windowName, bgr);
	} while (waitKey(30) < 0);

	return 0;
}
