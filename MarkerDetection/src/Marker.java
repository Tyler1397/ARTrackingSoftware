import org.opencv.core.Core;
import org.opencv.core.CvType;
import org.opencv.core.Mat;
import org.opencv.core.MatOfPoint2f;
import org.opencv.core.Size;
import org.opencv.imgcodecs.Imgcodecs;
import org.opencv.imgproc.Imgproc;

public class Marker {
	Mat bitArray;
	MatOfPoint2f points;
	Mat[] rotations;
	
	public Marker(){
		bitArray = new Mat();
		points = new MatOfPoint2f();
	}
	
	public Mat getMat(){
		return bitArray;
	}
	
	public void printMarker(){
		System.out.println(rotations[0].dump());
	}
	
	public Mat rotate(Mat in){
		//clone the input matrix
		Mat out = in.clone();	
		
		//rotate matrix clockwise, NOTE: matrix must be n x n
		for(int i = 0; i < in.rows();i++){
			for(int j = 0; j < in.cols();j++){
				out.put(i, j, in.get(in.cols()-j-1, i));
			}
		}
		
		// return rotated matrix
		return out;
	}
	
	public int hammDistMarker(Mat bits){
		Mat ids = new Mat( 4, 5, CvType.CV_8UC1 );
		int row = 0, col = 0;
		ids.put(row ,col, 1, 0,0,0,0,1,0,1,1,1,0,1,0,0,1,0,1,1,1,0);
		
		int dist = 0;
		
		for(int y = 0; y < 5; y ++){
			int minSum = Integer.MAX_VALUE;
			
			for(int p = 0; p < 4; p ++){	
				int sum = 0;				
				for(int x = 0; x < 5; x++){
					if(bits.get(y, x) == ids.get(p, x)){
						sum += 0;
					}else{
						sum += 1;
					}			
					//sum += (bits.get(y,x) == ids.get(p, x) ? 0 : 1);
				}
				if(minSum > sum){
					minSum = sum;
				}
			}
			dist += minSum;
		}
		return dist;
	}
	
	public int mat2id(Mat bits){
		int val = 0;
		for(int y = 0; y < 5; y++){
			val <<=1;
			if(bits.get(y,1) != null)
				val |=1;
			val <<=1;
			if(bits.get(y,1) != null)
				val |=1;
		}
		return val;
	}
	
	public int getMarkerId(Mat markerImage,int rotations){
		assert (markerImage.rows() == markerImage.cols());
		//assert(markerImage.type() == CvType.CV_8UC1 );
		
		Mat grey = markerImage;
		Imgproc.threshold(grey, grey, 125, 255, Imgproc.THRESH_BINARY);

		int cellSize = markerImage.rows() / 7;
		
		for(int y = 0; y < 7; y++){
			int inc = 6;
			
			if(y==0 || y == 6)
				inc = 1;
			
			for(int x = 0; x < 7; x+= inc){
				int cellX = x * cellSize;
				int cellY = y * cellSize;
				Mat cell = grey.submat(cellX, cellX+cellSize,cellY, cellY+cellSize);
				int nZ = Core.countNonZero(cell);
				if(nZ > (cellSize*cellSize /2)){
					return -1;
				}
				System.out.println("Valid");
			}
		}
		return -1;
	}
	
	public boolean equals(Object o){
		Marker temp = (Marker)o;
		for(int i = 0; i < 4; i ++){
				boolean same = true;
				for(int row = 0; row < temp.bitArray.rows(); row++){
					for(int col = 0; col < temp.bitArray.cols(); col++){
						if(this.rotations[0].get(row, col)[0] != temp.rotations[i].get(row, col)[0]){
							same = false;
						}
					}
				}
				if(same == true){
					return true;
				}
		}
		return false;
	}
	
	public void generateMarker(Mat bits, int n){	
		bitArray = new Mat(n,n,CvType.CV_8UC1);
		
		int cellSize = bits.width() / n;
		
		Mat grey = bits.clone();
		Imgproc.threshold(grey, grey, 125, 255, Imgproc.THRESH_BINARY);
		
		for(int col = 0; col < n; col ++){
			for(int row = 0; row < n; row ++){
				int cellX = row * cellSize;
				int cellY = col * cellSize;
				Mat cell = grey.submat(cellX, cellX+cellSize,cellY, cellY+cellSize);
				Core.extractChannel(cell, cell, 0);
				int zeros = 0;
				for(int i = 0; i < cell.rows(); i++){
					for(int j = 0; j < cell.cols();j++){
						if(cell.get(i, j)[0] == 0){
							zeros++;
						}
					}
				}
				int nZ = Core.countNonZero(cell);
				if(nZ > (cellSize * cellSize)/2){
					bitArray.put(row, col, 1);
				}else{
					bitArray.put(row, col, 0);
				}
			}
		}
		
		rotations = new Mat[4];
		Mat temp = bitArray.clone();
		for(int i = 0; i < 4 ; i ++){
			rotations[i] = temp.clone();
			temp = rotate(temp);	
		}
	}
}
