import java.util.ArrayList;
import java.util.LinkedList;

import org.opencv.core.*;
import org.opencv.imgcodecs.Imgcodecs;
import org.opencv.imgproc.Imgproc;
import org.opencv.imgproc.Moments;

public class Main {
	private static final String MARKER = "marker.png";
	private static final String SOURCE_IMAGE = "Research2.jpg";
	private static MatOfPoint2f m_markerCorners2d;
	private static int markerSize = 400;
	private static Marker mainMarker;
	
	public static void main(String[] args) {
		try {
			
			
			
			System.loadLibrary(Core.NATIVE_LIBRARY_NAME);
			Mat source = Imgcodecs.imread(SOURCE_IMAGE, Imgcodecs.CV_LOAD_IMAGE_COLOR);
			Mat mark = Imgcodecs.imread(MARKER, Imgcodecs.CV_LOAD_IMAGE_COLOR);
			Mat source2 = Imgcodecs.imread(SOURCE_IMAGE, Imgcodecs.CV_LOAD_IMAGE_COLOR);
		
			Mat destination = new Mat(source.rows(), source.cols(), source.type());

			destination = source;
			Imgproc.cvtColor(source, destination, Imgproc.COLOR_BGRA2GRAY);
			//Imgproc.blur(destination, destination, new Size(3, 3));
			Imgproc.threshold(destination, destination, 127, 255, Imgproc.THRESH_BINARY_INV);
			Imgproc.erode(destination, destination, Imgproc.getStructuringElement(Imgproc.MORPH_RECT, new Size(3, 3)),new Point(0, 0), 1);
			Imgproc.dilate(destination, destination, Imgproc.getStructuringElement(Imgproc.MORPH_RECT, new Size(3, 3)),new Point(0, 0), 1);

			mainMarker = new Marker();
			mainMarker.generateMarker(mark, 7);
			
			Imgcodecs.imwrite("ThreshZero.jpg", destination);
			ArrayList<MatOfPoint> contours = new ArrayList<MatOfPoint>();
			Mat hierarchy = new Mat();
			findContours(destination.clone(), contours, hierarchy, 90);
			
			ArrayList<Marker> possibleCanidates = findCanidates(contours,hierarchy);
			detectMarkers(destination.clone(),possibleCanidates);			
			
			Imgcodecs.imwrite("test2.png", source2);

		} catch (Exception e) {
			System.out.println("error: " + e.getMessage());
		}
	}

	// method takes in a threshold image, an empty array list , an empty matrix, and a minimum perimeter integer
	// and populates the contours and hierarchy, using the minimum perimieter variable to filter out unwanted contours
	public static void findContours(Mat thresholdImg, ArrayList<MatOfPoint> contours, Mat hierarchy,int minPerimeter) {
		Mat thresholdImgCopy = thresholdImg.clone();
		ArrayList<MatOfPoint> allContours = new ArrayList<MatOfPoint>();
		Imgproc.findContours(thresholdImgCopy, allContours, hierarchy, Imgproc.RETR_LIST, Imgproc.CHAIN_APPROX_SIMPLE);
		contours.clear();
		Mat outputImage = Imgcodecs.imread(SOURCE_IMAGE, Imgcodecs.CV_LOAD_IMAGE_COLOR);
		for (int i = 0; i < allContours.size(); i++) {
			MatOfPoint2f newMtx = new MatOfPoint2f(allContours.get(i).toArray());
			int contourSize = (int) Imgproc.arcLength(newMtx, true);
			if(contourSize > minPerimeter){
				contours.add(allContours.get(i));
			}
		}
	
		for (int i = 0; i < contours.size(); i++) {
			MatOfPoint2f newMtx = new MatOfPoint2f(contours.get(i).toArray());
			int contourSize = (int) Imgproc.arcLength(newMtx, true);
			if (contourSize > minPerimeter) {
				Imgproc.drawContours(outputImage, contours, i, new Scalar(0, 120, 0), 2, 8, hierarchy, 0, new Point());
			}
		}
		Imgcodecs.imwrite("DetectedContours.png", outputImage);
	}

	public static ArrayList<Marker> findCanidates(ArrayList<MatOfPoint> c, Mat hierarchy) {
		
		MatOfPoint2f approxCurve = new MatOfPoint2f();
		ArrayList<Marker> output = new ArrayList<Marker>();
		
		Mat outputImage = Imgcodecs.imread(SOURCE_IMAGE, Imgcodecs.CV_LOAD_IMAGE_COLOR);
		int timm = 1;
		for (MatOfPoint m : c) {
			MatOfPoint2f temp = new MatOfPoint2f(m.toArray().clone());
			approxCurve = new MatOfPoint2f();
			Imgproc.approxPolyDP(temp, approxCurve, Imgproc.arcLength(temp, true) * 0.02, true);

			if (approxCurve.total() != 4 || Imgproc.isContourConvex(m)) {
				continue;
			}
			
			Marker m1 = new Marker();
			
			for(int i = 0; i < 4; i ++){
				m1.points.push_back(new MatOfPoint2f(new Point(approxCurve.toList().get(i).x,approxCurve.toList().get(i).y)));
				Imgproc.circle(outputImage, new Point(approxCurve.toList().get(i).x,approxCurve.toList().get(i).y), 3, new Scalar(0, 0, 255), 10);
			}
			output.add(m1);
			Imgproc.drawContours(outputImage, c, c.indexOf(m), new Scalar(255 - timm * 20, 255, timm * 5), 2, 8, hierarchy, 0, new Point());
			timm++;
		}
		
		Imgcodecs.imwrite("Poly.png", outputImage);
		
		return output;
	}
	
	public static void detectMarkers(Mat grayscaleImg,ArrayList<Marker> canidates){
		m_markerCorners2d = new MatOfPoint2f();
		m_markerCorners2d.push_back(new MatOfPoint2f(new Point(0,0)));
		m_markerCorners2d.push_back(new MatOfPoint2f(new Point(markerSize-1,0)));
		m_markerCorners2d.push_back(new MatOfPoint2f(new Point(markerSize-1,markerSize-1)));
		m_markerCorners2d.push_back(new MatOfPoint2f(new Point(0,markerSize-1)));
		Mat test = new Mat();
		Mat outputImage = Imgcodecs.imread(SOURCE_IMAGE, Imgcodecs.CV_LOAD_IMAGE_COLOR);
		
		for(int i =0; i < canidates.size(); i ++){
			Marker marker = canidates.get(i);
			
			Mat m = Imgproc.getPerspectiveTransform(marker.points, m_markerCorners2d);
			Mat temp = m.clone();

			Imgproc.warpPerspective(grayscaleImg, test,m, new Size(markerSize,markerSize));
			Core.flip(test, test, 1);
			Imgproc.threshold(test, test, 127, 255, Imgproc.THRESH_BINARY_INV);
			marker.generateMarker(test,7);
			
			if(marker.equals(mainMarker)){
				System.out.println(Math.toDegrees(Math.atan2(temp.get(1, 0)[0], temp.get(0, 0)[0]))+" Marker "+ i);
				Imgcodecs.imwrite(i+"marker.png", test);

				Imgproc.line(outputImage, marker.points.toList().get(0),marker.points.toList().get(1), new Scalar(0, 255, 0), 3);
				Imgproc.line(outputImage, marker.points.toList().get(1),marker.points.toList().get(2), new Scalar(0, 255, 0), 3);
				Imgproc.line(outputImage, marker.points.toList().get(2),marker.points.toList().get(3), new Scalar(0, 255, 0), 3);
				Imgproc.line(outputImage, marker.points.toList().get(3),marker.points.toList().get(0), new Scalar(0, 255, 0), 3);
				Imgproc.putText(outputImage, "Marker "+i, new Point(marker.points.toList().get(3).x,marker.points.toList().get(3).y - 50), Core.FONT_ITALIC, .5, new Scalar(255, 255, 255));
			}else{
				Imgproc.line(outputImage, marker.points.toList().get(0),marker.points.toList().get(2), new Scalar(0, 0, 255), 3);
				Imgproc.line(outputImage, marker.points.toList().get(1),marker.points.toList().get(3), new Scalar(0, 0, 255), 3);
				Imgproc.line(outputImage, marker.points.toList().get(0),marker.points.toList().get(1), new Scalar(0, 0, 255), 3);
				Imgproc.line(outputImage, marker.points.toList().get(1),marker.points.toList().get(2), new Scalar(0, 0, 255), 3);
				Imgproc.line(outputImage, marker.points.toList().get(2),marker.points.toList().get(3), new Scalar(0, 0, 255), 3);
				Imgproc.line(outputImage, marker.points.toList().get(3),marker.points.toList().get(0), new Scalar(0, 0, 255), 3);
			}
		}
		Imgcodecs.imwrite("CorrectContour.jpg", outputImage);
	}
	
}
