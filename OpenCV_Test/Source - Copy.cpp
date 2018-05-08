#include <iostream>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/calib3d/calib3d.hpp>

using namespace std;
using namespace cv;

#define MIN_HAND_DEFECTS 4
#define MAX_HAND_DEFECTS 6
#define MIN_LENGTH_VARIATION 50
#define MAX_LENGTH_VARIATION 110
#define MIN_INNER_ANGLE 15
#define MAX_INNER_ANGLE 115

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

// Check if the hand defect lenghts are on an interval of 55% - 170% of a random length in a given unitCircle2
inline bool compareDefectsLength(const PHandDefect defects, vector<int>::size_type &i, const int offset, const vector<int> &unitCircle, const double *length)
{
	for (i = offset; i < unitCircle.size(); ++i)
	{
		if (defects[unitCircle[i]].length < length[0] ||
			defects[unitCircle[i]].length > length[1])
		{
			return false;
		}
	}
	return true;
}

// Add a HandDefect index to the corresponding unitCircle2 vector
inline bool addToUnitCircle2(const HandDefect &defect, const int index, vector<int> *unitCircle2)
{
	if (defect.unitCircle[0] == defect.unitCircle[1])
	{
		if (defect.unitCircle[0] == 0)
		{
			// {0, 1}
			// {0, 2}
			if (unitCircle2[0].size() < MIN_HAND_DEFECTS && unitCircle2[1].size() < MIN_HAND_DEFECTS)
			{
				unitCircle2[0].push_back(index);
				unitCircle2[1].push_back(index);
			}
			else
			{
				// abort, my image have more then 5 triangles per quadrant
				return false;
			}
		}
		else if (defect.unitCircle[0] == 1)
		{
			// {0, 1}
			// {3, 1}
			if (unitCircle2[0].size() < MIN_HAND_DEFECTS && unitCircle2[2].size() < MIN_HAND_DEFECTS)
			{
				unitCircle2[0].push_back(index);
				unitCircle2[2].push_back(index);
			}
			else
			{
				// abort, my image have more then 5 triangles per quadrant
				return false;
			}
		}
		else if (defect.unitCircle[0] == 2)
		{
			// {3, 2}
			// {0, 2}
			if (unitCircle2[3].size() < MIN_HAND_DEFECTS && unitCircle2[1].size() < MIN_HAND_DEFECTS)
			{
				unitCircle2[3].push_back(index);
				unitCircle2[1].push_back(index);
			}
			else
			{
				// abort, my image have more then 5 triangles per quadrant
				return false;
			}
		}
		else //if (defect.unitCircle[0] == 3)
		{
			// {3, 2}
			// {3, 1}
			if (unitCircle2[3].size() < MIN_HAND_DEFECTS && unitCircle2[2].size() < MIN_HAND_DEFECTS)
			{
				unitCircle2[3].push_back(index);
				unitCircle2[2].push_back(index);
			}
			else
			{
				// abort, my image have more then 5 triangles per quadrant
				return false;
			}
		}
	}
	else
	{
		static int sum;
		sum = defect.unitCircle[0] + defect.unitCircle[1];
		if (sum == 1)
		{
			// {0, 1}
			if (unitCircle2[0].size() < MIN_HAND_DEFECTS)
			{
				unitCircle2[0].push_back(index);
			}
			else
			{
				// abort, my image have more then 5 triangles per quadrant
				return false;
			}
		}
		else if (sum == 2)
		{
			// {0, 2}
			if (unitCircle2[1].size() < MIN_HAND_DEFECTS)
			{
				unitCircle2[1].push_back(index);
			}
			else
			{
				// abort, my image have more then 5 triangles per quadrant
				return false;
			}
		}
		else if (sum == 4)
		{
			// {3, 1}
			if (unitCircle2[2].size() < MIN_HAND_DEFECTS)
			{
				unitCircle2[2].push_back(index);
			}
			else
			{
				// abort, my image have more then 5 triangles per quadrant
				return false;
			}
		}
		else if (sum == 5)
		{
			// {3, 2}
			if (unitCircle2[3].size() < MIN_HAND_DEFECTS)
			{
				unitCircle2[3].push_back(index);
			}
			else
			{
				// abort, my image have more then 5 triangles per quadrant
				return false;
			}
		}
		else
		{
			// incorrect sum
			return false;
		}
	}
	return true;
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

// Returns the distance of 2 points
inline double distanceP2P(Point2d a, Point2d b)
{
	return sqrt(pow(a.x - b.x, 2) + pow(a.y - b.y, 2));
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

	// Combination of 2 quadrants
	// [0]: {0, 1}
	// [1]: {0, 2}
	// [2]: {3, 1}
	// [3]: {3, 2}
	vector<int> unitCircle2[4];
	unitCircle2[0].reserve(MIN_HAND_DEFECTS);
	unitCircle2[1].reserve(MIN_HAND_DEFECTS);
	unitCircle2[2].reserve(MIN_HAND_DEFECTS);
	unitCircle2[3].reserve(MIN_HAND_DEFECTS);

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
					double tmp;
					float radius;
					int i = 0, j = 0;

					vector<Point2f> farthestPoints;
					farthestPoints.reserve(listDefectsSize);

					// The convexity defects which have depth larger than a threshold value tend to appear around the palm portion
					for (; i < listDefectsSize; ++i)
					{
						tmp = defects[i][3] / 256.0;
						if (tmp > 60.0)
						{
							circle(bgr, contours[largestContour][defects[i][2]], 4, Scalar(0, 0, 255), 2);
							farthestPoints.push_back(contours[largestContour][defects[i][2]]);
						}
					}
					if (farthestPoints.size() > 3)
					{

					}

					minEnclosingCircle(farthestPoints, mCenter, radius); // returns the unit circle
					circle(bgr, mCenter, radius, Scalar(255, 0, 0));
					circle(bgr, mCenter, 4, Scalar(0, 255, 255), 2);

					int fingerIndex = 0;
					char sText[16] = {0};

					for (i = 0; i < listDefectsSize; ++i)
					{
						tmp = innerAngle(contours[largestContour][defects[i][0]], contours[largestContour][defects[i][1]], contours[largestContour][defects[i][2]]);
						if (tmp >= MIN_INNER_ANGLE &&
							tmp <= MAX_INNER_ANGLE)
						{
							listDefects[j].startPoint = contours[largestContour][defects[i][0]];
							listDefects[j].endPoint = contours[largestContour][defects[i][1]];
							listDefects[j].farthestPoint = contours[largestContour][defects[i][2]];
							listDefects[j].innerAngle = tmp;
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

							_itoa(fingerIndex++, sText, 10);
							putText(bgr, sText, listDefects[j].endPoint, CV_FONT_NORMAL, 0.5, Scalar(0, 0, 255));

							// Gets the unitCircle2 of the current HandDefect (using unitCircle[0] and unitCircle[1])
							if (addToUnitCircle2(listDefects[j], j, unitCircle2))
							{
								++j;
							}
							else
							{
								// Rejects the current image
								j = 0;
								break;
							}
						}
					}

					if (j >= MIN_HAND_DEFECTS && j <= MAX_HAND_DEFECTS)
					{
						bool found = false;
						double deltaLength[2] = {0};
						vector<int>::size_type l;

						for (i = 0; i < /*_countof(unitCircle2)*/4; ++i)
						{
							if (unitCircle2[i].size() == MIN_HAND_DEFECTS)
							{
								if (listDefects[unitCircle2[i].front()].length > listDefects[unitCircle2[i].back()].length)
								{
									deltaLength[0] = listDefects[unitCircle2[i].back()].length;
									deltaLength[1] = listDefects[unitCircle2[i].front()].length;
								}
								else
								{
									deltaLength[0] = listDefects[unitCircle2[i].front()].length;
									deltaLength[1] = listDefects[unitCircle2[i].back()].length;
								}

								// Check if the hand defects of the current unitCircle2 are between deltaLength[0] and deltaLength[1]
								if (compareDefectsLength(listDefects, l, 0, unitCircle2[i], deltaLength))
								{
									found = true;
									break;
								}
							}
						}

						if (found)
						{
							int index = 0;
							for (l = 0; l < unitCircle2[i].size(); ++l)
							{
								index = unitCircle2[i][l];
								circle(bgr, listDefects[index].startPoint, 4, Scalar(255, 0, 0), 2);
								circle(bgr, listDefects[index].endPoint, 4, Scalar(0, 255, 0), 2);
								circle(bgr, listDefects[index].farthestPoint, 4, Scalar(0, 0, 255), 2);
								line(bgr, listDefects[index].startPoint, listDefects[index].endPoint, Scalar(0, 255, 0), 1);
								line(bgr, listDefects[index].startPoint, listDefects[index].farthestPoint, Scalar(0, 255, 0), 1);
								line(bgr, listDefects[index].endPoint, listDefects[index].farthestPoint, Scalar(0, 255, 0), 1);
							}
						}

					}

					unitCircle2[0].clear();
					unitCircle2[1].clear();
					unitCircle2[2].clear();
					unitCircle2[3].clear();

				}
			}
		}

		imshow(windowName, bgr);
	} while (waitKey(30) < 0);

	return 0;
}
