/*
 * reservoir_mnist.cpp
 * 
 * Copyright 2015 Nils Schaetti <n.schaetti@gmail.com>
 * 
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 2 of the License, or
 * (at your option) any later version.
 * 
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 * 
 * You should have received a copy of the GNU General Public License
 * along with this program; if not, write to the Free Software
 * Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston,
 * MA 02110-1301, USA.
 * 
 * 
 */


#include <armadillo>
#include <stdlib.h>
#include <stdio.h>
#include <iostream>
#include <fstream>
#include <iostream>
#include <iomanip>
#include <getopt.h>
#include <string.h>
#include <fstream>
#include <dirent.h>
#include <sys/time.h>
#include <omp.h>

using namespace arma;
using namespace std;

#define ARMA_64BIT_WORD
#pragma GCC diagnostic ignored "-Wwrite-strings"

#define TIME_LOAD_IMAGES				0
#define TIME_COMPUTE_STATES				1
#define TIME_COMPUTE_X_TRANS			2
#define TIME_COMPUTE_YXT				3
#define TIME_COMPUTE_XXT				4
#define TIME_COMPUTE_INVERSE			5
#define TIME_COMPUTE_SR					6
#define TIME_LOAD_LABELS				7
#define TIME_COMPUTE_RESULTS			8
#define TIME_SOLVE						9
#define TIME_COMPUTE_TENSOR_PRODUCT		10
#define TIME_COUNT						11

/**
 * Time counter
 */
vec timeCounter = zeros<vec>(TIME_COUNT);

/**
 * Timer descriptions
 */
char* timerDesc[] = {	"Load images", 
						"Compute states",
						"Compute X transpose",
						"Compute YX^T matrix",
						"Compute XX^T matrix",
						"Compute inverse",
						"Compute spectral radius",
						"Load labels",
						"Resolve linear system",
						"Compute tensor product",
						"Compute results"};

/**
 * Time structures
 */
struct timeval time_start, time_end;
struct timeval timer_before, timer_after;

// Verbosity
int verbose = 0;

/*
 * Print the help
 * name: Usage
 * @param app_name Application's name
 * @param message Message to print
 * 
 */
void usage(char* app_name, char* message = "")
{
	#pragma omp master
	{
		fprintf(stderr, "Usage : %s <options>\n", app_name);
		fprintf(stderr, "%s\n", message);
		fprintf(stderr, "Options : \n");
		fprintf(stderr, " -t [--train] \t\t\t Number of training images\n");
		fprintf(stderr, " -e [--test] \t\t\t Number of test images\n");
		fprintf(stderr, " -i [--init] \t\t\t Number of initial images not used\n");
		fprintf(stderr, " -s [--seed] \t\t\t Initial seed for the random number generator\n");
		fprintf(stderr, " -n [--nothing] \t\t Space between images\n");
		fprintf(stderr, " -o [--size] \t\t\t Size of the reservoir\n");
		fprintf(stderr, " -p [--path] \t\t\t Path to the dataset\n");
		fprintf(stderr, " -a [--leaking-rate] \t Reservoir's leaking rate\n");
		fprintf(stderr, " -l [--spectral-radius] \t Reservoir's spectral radius\n");
	}
	exit(EXIT_FAILURE);
}

/*
 * Show reservoir properties
 * name: showProperties
 * @param trainLen Number of images in training set
 * @param testLen Number of images in test set
 * 
 */
void showProperties(int trainLen, int testLen, int initLen, int nothing, int reservoirSize, std::string datasetPath, int lenSize, int inSize, int outSize, float a, double spectralRadius, int totalTrainImages, int totalTestImages, double bias, double input_scaling, double input_sparsity, double reservoir_sparsity)
{
	#pragma omp master
	{
		printf("[INFO] Reservoir's properties : \n");
		printf("\tNumber of image(s) in training set : %i\n", trainLen);
		printf("\tNumber of image(s) in test set : %i\n", testLen);
		printf("\tNumber of init image(s) not used : %i\n", initLen);
		printf("\tSpace between images : %i\n", nothing);
		printf("\tTotal number of image(s) : %i\n", trainLen+testLen);
		printf("\tNumber of available training images : %i\n", totalTrainImages);
		printf("\tNumber of available testing images : %i\n", totalTestImages);
		printf("\tReservoir size (N_x) : %i\n", reservoirSize);
		printf("\tPath to the dataset : %s\n", datasetPath.c_str());
		printf("\tImages size : %i\n", lenSize);
		printf("\tNumber of inputs (N_u) : %i\n", inSize);
		printf("\tNumber of outputs (N_y) : %i\n", outSize);
		printf("\tLeaking rate (a) : %f\n", a);
		printf("\tSpectral radius : %f\n", spectralRadius);
		printf("\tBias : %f\n", bias);
		printf("\tInput scaling : %f\n", input_scaling);
		printf("\tInput sparsity : %f\n", input_sparsity);
		printf("\tReservoir sparsity : %f\n", reservoir_sparsity);
	}
}

/*
 * Print informations
 * name: printi
 * @param type Type of information as a string
 * @param chaine String to print
 */
void printfi(const char* type, const char* chaine)
{
	long mtime, seconds, useconds;
	
	gettimeofday(&time_end, NULL);
	
	seconds = time_end.tv_sec - time_start.tv_sec;
	useconds = time_end.tv_usec - time_start.tv_usec;
	
	mtime = ((seconds) * 1000 + useconds/1000.0) + 0.5;
	
	// only the master thread will execute this code
	#pragma omp master
	{
		printf("[%s][%ld ms] %s\n", type, mtime, chaine);
	}
}

/*
 * Print informations about a matrix
 * name: printfm
 * @param M Target matrix
 * @param name Matrix's name
 */
void printfm(mat* M, std::string name)
{
	printfi("INFO",(name + std::string(" : ") + std::to_string(M->n_rows) + std::string(" x ") + std::to_string(M->n_cols) + std::string(" columns")).c_str());
	if(verbose == 1)
		M->print(name);
	else if(verbose == 2)
		M->raw_print(name);
}

/*
 * Print informations about a matrix
 * name: printfm
 * @param M Target matrix
 * @param name Matrix's name
 */
void printfm(sp_mat* M, std::string name)
{
	printfi("INFO",(name + std::string(" : ") + std::to_string(M->n_rows) + std::string(" x ") + std::to_string(M->n_cols) + std::string(" columns")).c_str());
	if(verbose == 1)
		M->print(name);
	else if(verbose == 2)
		M->raw_print(name);
}

/*
 * Print timers informations
 * name: printfTimers
 */
void printft(void)
{
	// only the master thread will execute this code
	#pragma omp master
	{
		printf("\n");
	}
	double count = 0.0;
	for(int i=0; i<TIME_COUNT; i++)
		count += timeCounter(i);
	for(int i=0; i<TIME_COUNT; i++)
	{
		// only the master thread will execute this code
		#pragma omp master
		{
			printf("[TIME] %s : %f ms (%f percent)\n", timerDesc[i], timeCounter(i), (timeCounter(i)/count)*100.0);
		}
	}
}

void initTime(void)
{
	gettimeofday(&time_start, NULL);
}

/*
 * Start counting time
 * name: startTimer
 */
void startTimer(void)
{
	gettimeofday(&timer_before, NULL);
}

/*
 * Stop counting timer and add time
 * name: endTimer
 * @param timer Timer to add
 */
void endTimer(int timer)
{
	long mtime, seconds, useconds;
	
	// Get time of day
	gettimeofday(&timer_after, NULL);
	
	// Get seconds and milliseconds
	seconds = timer_after.tv_sec - timer_before.tv_sec;
	useconds = timer_after.tv_usec - timer_before.tv_usec;
	
	// Time
	mtime = ((seconds) * 1000 + useconds/1000.0) + 0.5;
	
	// Add time
	timeCounter(timer) += mtime;
}

/*
 * Load images properties
 * name: loadImageProperties
 * @param datasetPath Path to data
 * @param lenSize Image size
 * @param inSize Number of inputs
 * 
 */
void loadImageProperties(std::string descFile, int *lenSize, int *inSize)
{
	std::ifstream infile(descFile.c_str());
	std::string line;
	
	if(infile)
	{
		// Read
		getline(infile, line);
		*lenSize = atoi(line.c_str());
		getline(infile, line);
		*inSize = atoi(line.c_str()) * *lenSize;
	}
	else
	{
		// only the master thread will execute this code
		#pragma omp master
		{
			fprintf(stderr,"Can't open %s!\n", descFile.c_str());
		}
		exit(EXIT_FAILURE);
	}
}

/*
 * Returns the spectral radius
 * name: generateSpectralRadius
 * @param W Input matrix
 * @return Spectral radius as a double
 * 
 */
double generateSpectralRadius(sp_mat W)
{
	cx_vec eigval;
	cx_mat eigvec;
	
	// Get eigenvalue & vector
	eig_gen(eigval, eigvec, mat(W));
	
	return abs(eigval(0));
}

/*
 * Open an image and put the data in a matrix
 * name: loadImage
 * @param path Path to the image
 * @return Image matrix
 */
mat loadImage(std::string path)
{
	mat Mc1;
	int nb_lig1=1;
	int nb_col1=0;
	string ligne; // variable contenant chaque ligne lue 
	string buffer;
	double tampon;
	string Result; 
	
	// Open file read-only
	ifstream fichier(path.c_str(), ios::in);
	
	// Check opening
	if(fichier)
	{ 
		// Get number of columns
		getline(fichier, ligne);
		istringstream iss(ligne);
		while (iss >> buffer)
		{
			++nb_col1;
		}
		
		// Get number of lines
		while(std::getline(fichier,ligne))
		{
			nb_lig1++;
		}
		
		// Close the file
		fichier.close();
		
		// Re-open the file
		fichier.open (path.c_str(), ios::in);
		
		// Resize the matrix
		Mc1.resize(nb_lig1,nb_col1);
		
		// Load the file
		for (int i=0;i<nb_lig1;i++)
		{
			for (int j=0;j<nb_col1;j++)
			{
				fichier >> setprecision(75) >> tampon;
				Mc1(i,j) = tampon;
			}
		}
		
		// Close
		fichier.close();  
	}
	else 
	{
		cout << "[ERROR] Can't open image " << path.c_str() << "!" << endl;
		exit(EXIT_FAILURE);
	}
	
	// Info
	cout << "\r[INFO] " << path.c_str() << " loaded!";
	
	return Mc1;
}

/*
 * Rerturns the number of image in a directory
 * name: getImageCount
 * @param path Path to the image directory
 * @return Number of images
 */
int getImageCount(std::string path)
{
	int count = 0;
	DIR *dir;
	struct dirent *ent;
	
	// Open directory
	if((dir = opendir(path.c_str())) != NULL)
	{
		// All files
		while((ent = readdir(dir)) != NULL)
		{
			std::string filename(ent->d_name);
			if(filename.size() > 4)
			{
				if(filename.substr(filename.size()-4) == ".dat")
				{
					count++;
				}
			}
		}
		closedir(dir);
	}
	
	return count;
}

/*
 * Generate matrix Y containing target outputs
 * name: generateY
 * @param Yt Allocated Y matrix
 * @param trainLen Number of images in the training set
 * @param initLen Number of images not used in the training set
 * @param outSize Number of outputs
 * @param resultats Labels corresponding to the images
 * @param lenSize Images size
 * @return
 */
void generateY(sp_mat* Yt, int trainLen, int initLen, int outSize, vec resultats, int lenSize)
{
	int pos = 0;
	
	// On calcule la sortie
	for(int i = initLen; i < trainLen; i++)
	{
		colvec block = zeros<colvec>(10);
		//block *= -1;
		block((int)resultats(i)) = 1; 
		Yt->col(pos++) = block;
		//Yt((int)resultats(i),pos++) = 1;
	}
}

/*
 * Generate matrix X containing reservoir states
 * name: generateX
 * @param X Pointer to a allocated matrix
 * @param trainLen Number of image in the training set
 * @param initLen Number of image not used in the training set
 * @param lenSize Image size
 * @param inSize Number of inputs
 * @param resSize Reservoir size
 * @param Win Input matrix Win
 * @param W Reservoir matrix W
 * @param Wbias Bias matrix Wbias
 * @param datasetPath Path to dataset directory
 * @param a Leaky rate
 * @return
 * 
 */
void generateX(mat* X, int trainLen, int initLen, int lenSize, int inSize, int resSize, sp_mat Win, sp_mat W, mat Wbias, std::string datasetPath, double a)
{
	vec x = zeros<vec>(resSize);
	vec u(inSize);
	DIR *dir;
	struct dirent *ent;
	int count = 0;
	int t = 0;
	vec xx = zeros<vec>(resSize * lenSize);
	
	for(count = initLen+1; count <= initLen + trainLen; count++)
	{
		// Load image
		startTimer();
		mat im = loadImage(datasetPath + "/images/image_" + std::to_string(count) + ".dat");
		endTimer(TIME_LOAD_IMAGES);

		// Foreach column of the matrix
		startTimer();
		for(int k = 0; k < lenSize; k++)
		{
			// Get input
			u.subvec(0,inSize-1) = im.col(k);
			
			// Main equation
			x = (1-a)*x + a*tanh(Win*u + W*x + Wbias);
			
			// Put states in super-state
			X->submat(resSize*k, t, resSize*k+resSize-1, t) = x;
		}
		
		// Next step
		t++;
		
		endTimer(TIME_COMPUTE_STATES);
	}
}

/*
 * Generate matrix X containing reservoir states
 * name: generateX
 * @param X Pointer to a allocated matrix
 * @param trainLen Number of image in the training set
 * @param initLen Number of image not used in the training set
 * @param lenSize Image size
 * @param inSize Number of inputs
 * @param resSize Reservoir size
 * @param Win Input matrix Win
 * @param W Reservoir matrix W
 * @param Wbias Bias matrix Wbias
 * @param datasetPath Path to dataset directory
 * @param a Leaky rate
 * @return
 * 
 */
void generateShortcuts(mat* XXt, mat* YXt, int trainLen, int initLen, int lenSize, int inSize, int resSize, sp_mat Win, sp_mat W, mat Wbias, std::string datasetPath, double a, vec train_labels)
{
	vec x = zeros<vec>(resSize);
	vec u(inSize);
	DIR *dir;
	struct dirent *ent;
	int count = 0;
	int t = 0;
	vec xx = zeros<vec>(resSize * lenSize);
	int i,j;
	
	// For each image
	for(count = initLen+1; count <= initLen + trainLen; count++)
	{
		// Load image
		startTimer();
		mat im = loadImage(datasetPath + "/images/image_" + std::to_string(count) + ".dat");
		endTimer(TIME_LOAD_IMAGES);

		// Foreach column of the matrix
		startTimer();
		for(int k = 0; k < lenSize; k++)
		{
			// Get input
			u.subvec(0,inSize-1) = im.col(k);
			
			// Main equation
			x = (1-a)*x + a*tanh(Win*u + W*x + Wbias);
			
			// Put state in super-state
			xx.subvec(resSize*k, resSize*k+resSize-1) = x;
		}
		endTimer(TIME_COMPUTE_STATES);
		
		// Compute tensor product
		/*startTimer();
		mat k_xx = kron(xx,xx.t());
		endTimer(TIME_COMPUTE_TENSOR_PRODUCT);
		// Add to XX^t
		startTimer();
		*XXt += k_xx;
		endTimer(TIME_COMPUTE_XXT);*/
		
		startTimer();
		#pragma omp parallel for private(i,j) schedule(dynamic)
		for(i = 0; i < resSize * lenSize; i++)
		{
			for(j = i; j < resSize * lenSize; j++)
			{
				XXt->at(i,j) += xx(i)*xx(j);
				if(i!=j)
					XXt->at(j,i) += xx(i)*xx(j);
			}
		}
		endTimer(TIME_COMPUTE_XXT);
		
		// Add to YXt
		startTimer();
		YXt->row(train_labels(count-1)) += xx.t();
		endTimer(TIME_COMPUTE_YXT);
		
		// Next step
		t++;
	}
}

/*
 * 
 * name: generatePronostics
 * @param
 * @return
 * 
 */
mat generatePronostics(int testLen, int trainLen, int initLen, int lenSize, int inSize, int resSize, int outSize, sp_mat Win, sp_mat W, mat Wbias, mat Wout, std::string datasetPath, double a)
{
	vec x = zeros<vec>(resSize);
	vec u(inSize);
	int count = 0;
	int t = 0;
	double max = 0;
	int pos = 0;
	
	// Allocate memory for the super-state
	colvec superstate = zeros<colvec>(resSize*lenSize);
	
	// Pronostocs
	vec prono = zeros<vec>(testLen);
	
	// Info
	printfi("INFO","Computing reservoir states for testing...\n");
	
	for(count = 1; count <= testLen; count++)
	{
		// Load image
		startTimer();
		mat im = loadImage(datasetPath + "/test/image_" + std::to_string(count) + ".dat");
		endTimer(TIME_LOAD_IMAGES);
		
		// Foreach column of the matrix
		startTimer();
		vec maxs = zeros<vec>(outSize);
		for(int k = 0; k < lenSize; k++)
		{
			// Get input
			u.subvec(0,inSize-1) = im.col(k);
			
			// Main equation
			x = (1-a)*x + a*tanh(Win*u + W*x + Wbias);
			
			// Put states in super-state
			superstate.subvec(resSize*k, resSize*k+resSize-1) = x;
		}
		endTimer(TIME_COMPUTE_STATES);
		
		startTimer();
		
		// Compute the outputs
		vec output = Wout * superstate;
		
		// Find the maximum
		max = output(0);
		pos = 0;
		for(int y = 1; y < outSize; y++)
		{
			if(output(y) > max)
			{
				max = output(y);
				pos = y;
			}
		}
		prono(t) = pos;
		
		endTimer(TIME_COMPUTE_RESULTS);
		
		// Next step
		t++;
	}
	printfi("INFO","\n");
	
	return prono;
}

/*
 * Load the labels from dataset directory
 * name: loadLabels
 * @param path Path to dataset directory
 * @return Vector containing the labels
 * 
 */
vec loadLabels(std::string filename)
{
	vec Mc1;
	int nb_lig=1;
	std::string line;
	std::string buffer;
	double tampon;
	//std::string filename = path + "/labels";
	
	// Info
	#pragma omp master
	{
		cout << "[INFO] Loading labels from " << filename << endl;
	}
	
	// Open readonly file
	ifstream fichier(filename.c_str(), ios::in);
	
	// Opening ok?
	if(fichier)
	{
		// Get the number of linesch
		getline(fichier, line);
		istringstream iss(line);
		while(std::getline(fichier,line))
		{
			nb_lig++;
		}
		fichier.close();
		
		
		// Re-open a get data
		fichier.open (filename.c_str(), ios::in);
		Mc1.resize(nb_lig);
		for (int i=0;i<nb_lig;i++)
		{
			fichier>>tampon;
			Mc1(i)=tampon;
		}
		fichier.close();
	}
	else
	{
		#pragma omp master
		{
			cout << "[ERROR] Can't open " << filename << "!" << endl;
		}
		exit(0);
	}
	
	return Mc1;
}

/*
 * Generate random bias with scaling factor
 * name: generateBias
 * @param resSize Reservoir size
 * @param bias_scaling Bias' scaling factor
 * @return Bias matrix
 * 
 */
mat generateBias(int resSize, double bias_scaling)
{
	mat Wbias = randu(resSize, 1) - 0.5;
	return Wbias * bias_scaling;
}

/*
 * Generate random input matrix
 * name: generateWin
 * @param resSize Reservoir size
 * @param inSize Number of inputs
 * @param input_sparsity Input sparsity
 * @return Input matrix Win
 */
sp_mat generateWin(int resSize, int inSize, double input_sparsity)
{
	sp_mat Win = sprandu<sp_mat>(resSize, inSize,1.0-input_sparsity);
	for(int i=0; i<resSize; i++)
	{
		for(int j=0; j<inSize; j++)
		{
			if(Win(i,j) != 0.0)
			{
				if(((double)rand() / (double)RAND_MAX) <= 0.5)
				{
					Win(i,j) *= -1.0;
				}
			}
		}
	}
	
	return Win;
}

/*
 * 
 * name: inconnu
 * @param
 * @return
 * 
 */
sp_mat generateW(int resSize, double reservoir_sparsity)
{
	sp_mat W = sprandu<sp_mat>(resSize,resSize,1.0-reservoir_sparsity);
	for(int i=0; i<resSize; i++)
	{
		for(int j=0; j<resSize; j++)
		{
			if(W(i,j) != 0.0)
			{
				if(((double)rand() / (double)RAND_MAX) <= reservoir_sparsity)
				{
					W(i,j) *= -1.0;
				}
			}
		}
	}
	return W;
}

/*
 * Train Wout from X and Y matrices
 * name: trainWout
 * @param X Pointer to matrix X
 * @param Y Pointer to matrix Y
 * @return Trained Wout matrix
 */
mat trainWout(mat *X, sp_mat *Y, int use_solve)
{
	double reg = 1e-8;
	mat Wout;
	
	// X tranpose
	printfi("INFO","Calculating X transpose...");
	startTimer();
	mat X_T = X->t();
	endTimer(TIME_COMPUTE_X_TRANS);
	
	// YX_T matrix
	printfi("INFO","Calculating YX^T...");
	startTimer();
	mat YX_T = (*Y * X_T);
	endTimer(TIME_COMPUTE_YXT);
	printfm(&YX_T,"YX_T");
	//YX_T.print("YX_T : ");
	
	// XX_T matrix
	printfi("INFO","Calculating XX^t...");
	startTimer();
	mat XX_T = (*X * X_T);
	endTimer(TIME_COMPUTE_XXT);
	printfm(&XX_T,"XX_T");
	X->resize(0,0);
	//XX_T.print("XX_T : ");
	
	// Inverse OR resolve
	if(!use_solve)
	{
		printfi("INFO","Calculating inverse...");
		startTimer();
		mat inv_XX_T = inv(XX_T);
		endTimer(TIME_COMPUTE_INVERSE);
		printfm(&inv_XX_T,"XX_T_-1");
		
		// Wout
		Wout = YX_T * inv_XX_T;
	}
	else
	{
		printfi("INFO","Solving system of linear equations...");
		startTimer();
		Wout = solve(XX_T,YX_T.t());
		endTimer(TIME_SOLVE);
		Wout = Wout.t();
		printfm(&Wout,"Wout : ");
	}
	
	return(Wout);
}

/*
 * Train Wout from X and Y matrices
 * name: trainWout
 * @param X Pointer to matrix X
 * @param Y Pointer to matrix Y
 * @return Trained Wout matrix
 */
mat trainWoutShortcuts(mat *XXt, mat *YXt, int use_solve)
{
	double reg = 1e-8;
	mat Wout;
	
	// Inverse OR resolve
	if(!use_solve)
	{
		printfi("INFO","Calculating inverse...");
		startTimer();
		mat inv_XX_T = inv(*XXt);
		endTimer(TIME_COMPUTE_INVERSE);
		printfm(&inv_XX_T,"XX_T_-1");
		
		// Wout
		Wout = *YXt * inv_XX_T;
	}
	else
	{
		printfi("INFO","Solving system of linear equations...");
		startTimer();
		Wout = solve(*XXt,YXt->t());
		endTimer(TIME_SOLVE);
		Wout = Wout.t();
	}
	
	// Free memory
	XXt->resize(0,0);
	YXt->resize(0,0);
	
	printfm(&Wout,"Wout : ");
	return(Wout);
}

double getErrorRate(vec testLabels, vec prono, int testLen)
{
	double count = 0.0;
	
	for(int i=0; i<testLen; i++)
	{
		if(testLabels(i) != prono(i))
		{
			count++;
		}
	}
	
	return count/(double)testLen*100.0;
}

/*
 * Main function (we start here)
 * name: main
 * @param argc Number of arguments
 * @param argv Array of arguments
 * @return Exit state
 * 
 */
int main(int argc, char *argv[])
{
	// Reservoir properties
	int trainLen = 0;										// Number of images in training set
	int testLen = 0;										// Number of images in test set
	int initLen = 0;										// Number of initial images not used
	int beginLen = 0;
	int seed = 91;
	int nothing = 0;
	int resSize = 0;
	double a = 0.3;
	double rhoW;
	double spectralRadius = 0.9;
	int totalTrainCount = 0;
	int totalTestCount = 0;
	double bias = 0.0;
	double input_scaling = 1.0;
	double input_sparsity = 0.0;
	double reservoir_sparsity = 0.0;
	int xxt_shortcut = 0;
	int use_solve = 0;
	mat X;
	sp_mat Y;
	mat XXt;
	mat YXt;
	mat Wout;
	int threadid=0;											// id of thread
	int nthreads=0;											// number of thread available
	
	// Arguments variables
	int c;													// Argument
	int option_index = 0;
	static struct option long_options[] = {					// Long options
		{"train",  1, 0, 't'},
		{"test", 1, 0, 'e'},
		{"init", 1, 0, 'i'},
		{"seed", 1, 0, 's'},
		{"nothing", 1, 0, 'n'},
		{"reservoir-size", 1, 0, 'o'},
		{"path", 1, 0, 'p'},
		{"leaking-rate", 1, 0, 'a'},
		{"spectral-radius", 1, 0, 'l'},
		{"bias-scaling", 1, 0, 'b'},
		{"input-scaling", 1, 0, 'c'},
		{"input-sparsity", 1, 0, 'r'},
		{"reservoir-sparsity", 1, 0, 'y'},
		{"xxt-shortcut", 0, 0, 'x'},
		{"use-solve", 0, 0, 'd'}
	};
	std::string option;
	
	// Data properties
	std::string datasetPath = "";
	int lenSize = 0;										// Image size
	int inSize = 0;											// Number of inputs
	int outSize = 10;										// Numner of outputs
	
	// Foreach arguments
	while((c = getopt_long(argc, argv, "t:e:i:s:n:i:s:o:p:a:l:b:c:r:y:dxhv", long_options, &option_index)) != -1)
	{
		switch(c)
		{
			// Long option
			case 0:
				// Option's string
				option = long_options[option_index].name;

				// Training length
				if(option == "train")
				{
					trainLen = atoi(optarg);
				}
				// Test length
				else if(option == "test")
				{
					testLen = atoi(optarg);
				}
				// Init length
				else if(option == "init")
				{
					initLen = atoi(optarg);
				}
				// Space
				else if(option == "nothing")
				{
					nothing = atoi(optarg);
				}
				// Reservoir's size
				else if(option == "reservoir-size")
				{
					resSize = atoi(optarg);
				}
				// Dataset's path
				else if(option == "path")
				{
					datasetPath = std::string(optarg);
				}
				// Leaking rate
				else if(option == "leaking-rate")
				{
					a = atof(optarg);
				}
				// Spectral radius
				else if(option == "spectral-radius")
				{
					spectralRadius = atof(optarg);
				}
				// Bias
				else if(option == "bias-scaling")
				{
					bias = atof(optarg);
				}
				// Input scaling
				else if(option == "input-scaling")
				{
					input_scaling = atof(optarg);
				}
				// Seed
				else if(option == "seed")
				{
					seed = atoi(optarg);
				}
				// Input sparsity
				else if(option == "input-sparsity")
				{
					input_sparsity = atof(optarg);
				}
				// Reservoir sparsity
				else if(option == "reservoir-sparsity")
				{
					reservoir_sparsity = atof(optarg);
				}
				// XXt shortcut
				else if(option == "xxt_shortcut")
				{
					xxt_shortcut = true;
				}
				// Solve
				else if(option == "solve")
				{
					use_solve = 1;
				}
				break;
			// Training length
			case 't':
				trainLen = atoi(optarg);
				break;
			// Test length
			case 'e':
				testLen = atoi(optarg);
				break;
			// Init
			case 'i':
				initLen = atoi(optarg);
				break;
			// Nothing
			case 'n':
				nothing = atoi(optarg);
				break;
			// Reservoir size
			case 'o':
				resSize = atoi(optarg);
				break;
			// Dataset's path
			case 'p':
				datasetPath = std::string(optarg);
				break;
			// Leaking rate
			case 'a':
				a = atof(optarg);
				break;
			// Spectral radius
			case 'l':
				spectralRadius = atof(optarg);
				break;
			// Bias
			case 'b':
				bias = atof(optarg);
				break;
			// Input scaling
			case 'c':
				input_scaling = atof(optarg);
				break;
			// Help
			case 'h':
				usage(argv[0]);
				break;
			// Input sparsity
			case 'r':
				input_sparsity = atof(optarg);
				break;
			// Reservoir sparsity
			case 'y':
				reservoir_sparsity = atof(optarg);
				break;
			// Use solve
			case 'd':
				use_solve = true;
				break;
			// Verbose
			case 'v':
				verbose++;
				break;
			// XXt shortcut
			case 'x':
				xxt_shortcut = true;
				break;
			default:
				usage(argv[0]);
				break;
		}
	}
	
	// Time initialisation
	initTime();
	
	// Check reservoir size
	if(resSize <= 0)
		usage(argv[0], "[ERROR] The reservoir size must be greater than zero!!\n");
		
	// Check dataset path
	if(datasetPath == "")
		usage(argv[0], "[ERROR] I need to know where are the images!!\n");
	
	// Information about parallelization
	#pragma omp parallel private(threadid)
	{
		// only the master thread will execute this code
		#pragma omp master
		{
			// computation of the total number of threads
			nthreads=omp_get_num_threads();
			cout<<endl<<nthreads<<" thread(s) available for computation"<<endl;
		}
		
		// barrier to display nthreads before threadid
		#pragma omp barrier

		threadid=omp_get_thread_num();
		// in order to "protect" the common output
		#pragma omp critical
		{
			cout<<"Thread "<<threadid<<" is ready for computation"<<endl;
		}
	}
	
	// Load images properties
	printfi("INFO","Load images properties...");
	loadImageProperties(datasetPath + "/desc", &lenSize, &inSize);
	
	// Initialize random number generator
	printfi("INFO", "Initialize random number generator...");
	arma_rng::set_seed(seed);
	srand(time(NULL));
	
	// Input matrix
	sp_mat Win = generateWin(resSize, inSize, input_sparsity);
	printfm(&Win, "Win");
	
	// Reservoir matrix
	//mat W = randu(resSize,resSize) - 0.5;
	sp_mat W = generateW(resSize, reservoir_sparsity);
	printfm(&W, "W");
	
	// Spectral radius
	startTimer();
	rhoW = generateSpectralRadius(W);
	printfi("INFO",(std::string("Largest eigenvalue for W is ") + std::to_string(rhoW)).c_str());
	W = W * (spectralRadius / rhoW);
	endTimer(TIME_COMPUTE_SR);
	printfm(&W, "W after spectral radius");
	
	// Get total number of images
	totalTrainCount = getImageCount(datasetPath + std::string("/images"));
	totalTestCount = getImageCount(datasetPath + std::string("/test"));
	
	// Check image count
	if(trainLen > totalTrainCount || testLen > totalTestCount)
	{
		printfi("ERROR","Not enough image available!");
		exit(EXIT_FAILURE);
	}
	
	// Show reservoir properties
	showProperties(trainLen, testLen, initLen, nothing, resSize, datasetPath, lenSize, inSize, outSize, a, spectralRadius, totalTrainCount, totalTestCount, bias, input_scaling, input_sparsity, reservoir_sparsity);
	
	// Load labels
	startTimer();
	vec train_labels = loadLabels(datasetPath + std::string("/train_labels"));
	vec test_labels = loadLabels(datasetPath + std::string("/test_labels"));
	endTimer(TIME_LOAD_LABELS);
	
	// Get bias matrix
	printfi("INFO","Generating Wbias...");
	mat Wbias = generateBias(resSize, bias);
	printfm(&Wbias, "Wbias");
	
	// Get states matrix
	if(!xxt_shortcut)
	{
		printfi("INFO","Computing reservoir states...");
		X = zeros<mat>(resSize * lenSize, trainLen);
		generateX(&X, trainLen, initLen, lenSize, inSize, resSize, Win, W, Wbias, datasetPath, a);
		printfi("INFO","\n");
		printfm(&X, "X");
	}
	else
	{
		printfi("INFO","Computing XX^t and YX^t...");
		XXt = zeros<mat>(resSize * lenSize, resSize * lenSize);
		YXt = zeros<mat>(outSize, resSize * lenSize);
		generateShortcuts(&XXt, &YXt, trainLen, initLen, lenSize, inSize, resSize, Win, W, Wbias, datasetPath, a, train_labels);
		printfm(&XXt, "XXt");
		printfm(&YXt, "YXt");
	}
	
	// Get target matrix (if necessary)
	if(!xxt_shortcut)
	{
		printfi("INFO","Generating target matrix...");
		Y = zeros<sp_mat>(outSize,trainLen);
		generateY(&Y, trainLen, initLen, outSize, train_labels, lenSize);
		printfm(&Y, "Y");
	}
	
	// Train the output
	printfi("INFO","Starting training phase...");
	if(!xxt_shortcut)
		Wout = trainWout(&X,&Y,use_solve);
	else
		Wout = trainWoutShortcuts(&XXt, &YXt, use_solve);
	printfm(&Wout, "Wout");
	
	// Test the reservoir
	printfi("INFO","Starting testing phase...");
	vec pronostics = generatePronostics(testLen, trainLen, initLen, lenSize, inSize, resSize, outSize, Win, W, Wbias, Wout, datasetPath, a);
	
	// Get error rate
	#pragma omp master
	{
		printf("[END] Error rate : %f\n", getErrorRate(test_labels, pronostics, testLen));
	}
	
	// Print time analysis
	printft();
	
	// End
	exit(EXIT_SUCCESS);
}
