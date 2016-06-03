#include <armadillo>
#include <stdlib.h>
#include <stdio.h>
using namespace arma;
#include <iostream>
#include <fstream>
using namespace std;

#include <iostream>     // std::cout, std::fixed
#include <iomanip>  

const int lenSize=20;
const int inSize =80*2;

string path= "../datasets/MNIST/test1/";

//string path= "/home/couturie/ajeter/reservoir/NILS/Generator/standard_big/";


extern "C" {
  void openblas_set_num_threads(int num_threads);
  }



mat conv2d(const mat& sample, const mat &kernel ) {

  int x_s = sample.n_rows, x_k = kernel.n_rows;
  int y_s = sample.n_cols, y_k = kernel.n_cols;
  int x_o = x_s + x_k - 1, y_o = y_s + y_k - 1;
  mat output = zeros<mat>( x_o, y_o);   // Must explicitly zero out Eigen matrices


  for (int row = 0; row < x_s; row++) {
    for (int col = 0; col < y_s; col++) {
      output.submat(row, col, row+x_k-1, col+y_k-1) += sample(row, col) * kernel;
    }
  }


/*    cout<<"ici"<<endl<<output.rows()<<endl;
      Matrix<double,Dynamic,Dynamic> res=output;
      cout<<"ici"<<endl<<res.rows()<<endl;
*/
  double max=output.max();
  return output.submat(x_k/2, y_k/2, x_k/2+x_s-1, y_k/2+y_s-1)/max;
}




//Function to load the input files
mat load_input(string path, int Number){
  mat  Mc1(inSize,lenSize);

  string ligne; // variable contenant chaque ligne lue 
  string buffer;
  double tampon;
  string Result; 
  ostringstream convert;   
  convert << Number;      
  Result = convert.str(); 


  path=path+"image_"+Result+".dat";

//  cout<<path<<endl;
  
  ifstream fichier(path.c_str(), ios::in);  // on ouvre le fichier en lecture
  if ( fichier ) // ce test échoue si le fichier n'est pas ouvert 
  { 
 
    for (int i=0;i<inSize;i++){
      for (int j=0;j<lenSize;j++){	
	fichier>>setprecision(10)>>tampon;
	Mc1(i,j)=tampon;	
      }
    }

//    cout<<size(Mc1)<<endl;
			 
    fichier.close();  
				
  } // on ferme le fichier
  else {  
    cout << "Impossible d'ouvrir le fichier 1!" << endl;
    exit(0);
  }
  return Mc1;
}







//Function to load the label files
vec load_label(string path){   //New function
  vec Mc1;
  int nb_lig=1;
  int nb_col=0;
  string ligne; // variable contenant chaque ligne lue 
  string buffer;
  double tampon;


  cout<<"ici"<<path<<endl;
  
  ifstream fichier(path.c_str(), ios::in);  // on ouvre le fichier en lecture
  if ( fichier ) // ce test échoue si le fichier n'est pas ouvert 
  { 
    getline(fichier, ligne);
    istringstream iss(ligne);
      
    while ( std::getline( fichier, ligne ) ) 
    { 
      nb_lig++;
    }
			
    fichier.close();
  
			
    fichier.open (path.c_str(), ios::in);
    Mc1.resize(nb_lig);
				
    for (int i=0;i<nb_lig;i++){
      fichier>>tampon;
      Mc1(i)=tampon;	
    }

			 
    fichier.close();  
				
  } // on ferme le fichier
  else {  
    cout << "Impossible d'ouvrir le fichier 1!" << endl;
    exit(0);
  }
  return Mc1;
}




int main( int argc, char *argv[] ) {


  

  int trainLen = 500;
  int testLen = 20;
  int initLen = 50;
  int beginLen = 0;
  int seed=91;
  int nothing=0;
//size of the reservoir
  int resSize = 500;
  double a = 0.3; // leaking rate  0.3
  int nb_times=1;

  
  for(int i=1;i<argc;i++) {
    if (!strcmp(argv[i],"-train")) {
      trainLen=atoi(argv[i+1]);
      i++;
    }
    else if (!strcmp(argv[i],"-test")) {
      testLen=atoi(argv[i+1]);
      i++;
    }
    else if (!strcmp(argv[i],"-init")) {
      initLen=atoi(argv[i+1]);
      i++;
    }
    else if (!strcmp(argv[i],"-leaky")) {
      a=atof(argv[i+1]);
      i++;
    }
    else if (!strcmp(argv[i],"-seed")) {
      seed=atoi(argv[i+1]);
      i++;
    }
    else if (!strcmp(argv[i],"-ressize")) {
      resSize=atoi(argv[i+1]);
      i++;
    }
    else if (!strcmp(argv[i],"-nbtimes")) {
      nb_times=atoi(argv[i+1]);
      i++;
    }
    else {
      cout<<"error"<<endl;
      exit(0);
    }
  }

  cout<<"leaky "<<a<<endl;
  cout<<"init "<<initLen<<endl;
  cout<<"train "<<trainLen<<endl;
  cout<<"test "<<testLen<<endl;
  cout<<"ressize "<<resSize<<endl;
  cout<<"seed "<<seed<<endl;
  cout<<"nbtimes "<<nb_times<<endl;

  

  

  int RESULT[testLen];
  for(int i=0;i<testLen;i++)
    RESULT[i]=0;

  

  const int outSize = 10;

   arma_rng::set_seed(seed++);


  for(int nb_t=0;nb_t<nb_times;nb_t++) {


 
    
    //on garde uniquement 3 états, les 3 suivants
    int nb_state=5;
    int state1=6;
    int state2=9;
    int state3=12;
    int state4=15;
    int state5=18;

  
    mat Win = randu(resSize,1+inSize)-0.5;


    //matrice W
    mat W = randu(resSize,resSize)-0.5;

  
/*   //par bloc 
     mat W=zeros<mat>(resSize,resSize);

     int blocksize=20;
     for(int i=0;i<resSize/blocksize;i++) {
     W.submat(blocksize*i,blocksize*i,blocksize*(i+1)-1,blocksize*(i+1)-1)=randu(blocksize,blocksize)-0.5;
     }
*/

/*   mat W=zeros<mat>(resSize,resSize);
     srand(seed);
     int size=resSize;
     int ind=0;
     do {
     int s=lrand48()%10+15;
     if(s>size) {
     s=size;
     }
     cout<<s<<" "<<size<<endl;
     size-=s;
     W.submat(ind,ind,ind+s-1,ind+s-1)=randu(s,s)-0.5;
     ind+=s;
     }
     while(size>0);
*/

  
/*  //avec 3 diagonales
    mat W=zeros<mat>(resSize,resSize);
    W.diag()=randu(resSize)-0.5;

    vec temp=randu(resSize)-0.5;
    vec temp2=randu(resSize)-0.5;
    for(int i=0;i<resSize-1;i++)
    W(i,i+1)=temp(i);
    W(resSize-1,0)=temp(resSize-1);

    for(int i=0;i<resSize-2;i++)
    W(i,i+2)=temp2(i);
    W(resSize-2,0)=temp2(resSize-2);
    W(resSize-1,1)=temp2(resSize-1);
*/ 
  
    cout<<"Computing spectral radius..."<<endl;

  
    cx_vec eigval;
    cx_mat eigvec;
    eig_gen(eigval, eigvec,W);
    double rhoW=abs(eigval(0));

    cout<<"largest eigenvalue Win "<<rhoW<<endl;
    W = W * 1.25 / rhoW;  //1.25


    //allocated memory for the design (collected states) matrix
    mat X = zeros<mat>(1+resSize*nb_state,trainLen);
    //set the corresponding target matrix directly
    mat Yt = zeros<mat>(outSize,trainLen);

    //on mélange les images de training


    vec resultat=load_label(path+"train_labels");

    //on calcule la sortie
    for(int i=0;i<trainLen;i++) {
    
      colvec block=zeros<colvec>(10);
    
      block(resultat(i))=1; 
    
    
      Yt.col(i)=block;
      
    }

  

    vec null=zeros<vec>(inSize);
  
  
    //run the reservoir with the data and collect X
 


    mat matConv=ones<mat>(5,5);
    matConv.col(2)=ones<colvec>(5)*2;
    matConv.row(2)=ones<rowvec>(5)*2;
    matConv(2,2)=3;


    //training of the reservoir

    openblas_set_num_threads(1);
  
#pragma omp parallel for shared(X) 
    for (int t=0;t<trainLen; t++) {

      //x and u are local
      vec x=zeros<vec>(resSize);
      vec u(1+inSize);
      u(0)=1;     //1 always
    
      int index=t+1;  //+1 car ca commence a 1
      mat im=load_input(path+"images/",index);
      //im=conv2d(im,matConv);


//    cout<<t<<endl;

      x=zeros<vec>(resSize);
      for (int k=0;k<lenSize;k++) {
	u.subvec(1,inSize)=im.col(k);

	x = (1-a)*x + a*tanh( Win * u + W * x );

	if (k==state1 || k==state2 || k==state3 || k==state4 || k==state5) {
	  X(0,t)=1;
	  int k2=-1;
	  if(k==state1)
	    k2=0;
	  if(k==state2) 
	    k2=1;
	  if(k==state3)
	    k2=2;
	  if(k==state4) 
	    k2=3;
	  if(k==state5)
	    k2=4;
	
	  X.submat(1+resSize*k2,t,1+resSize*(k2+1)-1,t) = x;
	
	}
      }


      //on rajoute éventuellement du blanc
      for(int k=lenSize;k<lenSize+nothing;k++) {
	u.subvec(1,inSize)=null;
	x = (1-a)*x + a*tanh( Win * u + W * x );
      }
    }

    cout<<"end of images reading"<<endl;
    openblas_set_num_threads(8);

    // train the output
    double reg = 1e-8;

    mat X_T = X.t();
    mat A=(X * X_T);
    X.resize(0,0);

    A.diag()+=reg;

    mat B=(Yt * X_T).t();
    Yt.resize(0,0);
    X_T.resize(0,0);

    cout<<"start solve"<<endl;
  
    mat Wout = solve( A , B );


    cout<<"end of regression"<<endl;
  

    A.resize(0,0);
    B.resize(0,0);
  

  
    int nb_error=0;


    resultat=load_label(path+"test_labels");

  
    //testing the new images

    openblas_set_num_threads(1);
#pragma omp parallel for shared(X)  reduction(+:nb_error)
    for (int t=0;t<testLen;t++){

  
      int val=t+1;   //+1 car ca commence a 1
    
      mat im=load_input(path+"test/",val);
      //im=conv2d(im,matConv);

      vec x=zeros<vec>(resSize);
      vec u(1+inSize);
      u(0)=1;     //1 always
      vec newState(1+resSize*nb_state);
      newState(0)=1; //1 always

      for (int k=0;k<lenSize;k++) {
	u.subvec(1,inSize)=im.col(k);
	x = (1-a)*x + a*tanh( Win * u + W * x );


	if (k==state1 || k==state2 || k==state3 || k==state4 || k==state5) {
	  int k2=-1;
	  if(k==state1)
	    k2=0;
	  if(k==state2)
	    k2=1;
	  if(k==state3)
	    k2=2;
	  if(k==state4)
	    k2=3;
	  if(k==state5)
	    k2=4;
	
	  newState.subvec(1+resSize*k2,1+resSize*(k2+1)-1)=x;
	}


      }

      mat tmp = Wout.t()*newState;
      vec y=tmp.col(0);
      colvec new_block=y;

     
      int label=resultat(val-1);

      uword pronostic=0;
      double max_v=new_block.max(pronostic);


      //calcul du pronostic
      if(label!=(int)pronostic) {
//         nb_error++;
//      cout<<"on cherche image n°"<<val<<"   label "<<label<<endl;
//      cout<<" prono "<<pronostic<<endl;
	//     cout<<endl<<endl<<new_block<<endl;
#pragma omp critical
	RESULT[val-1]++;
      }


      //on rajoute éventuellement du blanc
      for(int k=0;k<nothing;k++) {
	u.subvec(1,inSize)=null;

	x = (1-a)*x + a*tanh( Win * u + W * x );
      
      
      
      }
    
    }


  }

  int nb_error=0;
  for(int i=0;i<testLen;i++)
    if(RESULT[i]>=nb_times/2.)
      nb_error++;

  
  cout<<endl<<"Error = "<<nb_error*100./testLen<<endl;

  
}


