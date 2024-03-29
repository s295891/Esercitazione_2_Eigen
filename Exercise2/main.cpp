#include <iostream>
#include "Eigen/Eigen"

using namespace std;
using namespace Eigen;

bool SolveLUQRdecomposition(const MatrixXd& A,
                            const VectorXd& b,
                            const VectorXd& exactSolution,
                            double& detA,
                            double& condA,
                            double& errore_relativo_lu,
                            double& errore_relativo_qr){
    JacobiSVD<MatrixXd> svd(A);
    VectorXd singularValuesA = svd.singularValues();
    condA = singularValuesA.maxCoeff() / singularValuesA.minCoeff();

    detA = A.determinant();

    if( singularValuesA.minCoeff() < 1e-16){
        return false;
    }
    else{
        VectorXd x_lu = A.fullPivLu().solve(b);
        errore_relativo_lu = (x_lu-exactSolution).norm() / exactSolution.norm();
        VectorXd x_qr = A.fullPivHouseholderQr().solve(b);
        errore_relativo_qr = (x_qr-exactSolution).norm() / exactSolution.norm();

        return true;
    }
}


int main()

{
    VectorXd exactSolution {{-1.0e+0, -1.0e+00}};


    {
        //_____________________-PRIMA MATRICE E VETTORE___________________
        MatrixXd A { {5.547001962252291e-01, -3.770900990025203e-02},
            {8.320502943378437e-01, -9.992887623566787e-01}
        };

        VectorXd b {{-5.169911863249772e-01, 1.672384680188350e-01}};

        double condA,detA,errore_relativo_lu, errore_relativo_qr;



        //    FullPivLU<Matrix2d> lu1(A1);
        //    Matrix2d P1 = lu1.permutationP();



        if(SolveLUQRdecomposition(A,b,exactSolution, detA, condA, errore_relativo_lu, errore_relativo_qr)){
            cout<< "La Prima matrice A : " <<endl<< A<<endl << " non e' singolare, con determinante "<<endl << detA<< endl << " e numero di condizionamento : " <<endl<< condA <<endl;
            cout<< "La soluzione del sistema con metodo di fattorizzazione LU, rispetto alla soluzione esatta "<<endl << exactSolution <<endl << "restituisce un errore relativo pari a "<<endl << errore_relativo_lu<<endl;
            cout << "La soluzione con fattorizzazione qr da un errore relativo pari a "<<endl << errore_relativo_qr << endl;
        }
        else{
            cout << "La Prima matrice e' singolare" << endl;
        }


        //     FullPivHouseholderQR<Matrix2d> qr1(A1);
        //     Matrix2d Q1 = qr1.matrixQ();
        //    Matrix2d R1_qr = Q1.transpose() * A1;
        //    cout << "Matrice QR : " << Q1 << " e la matrice R :  "<< R1<< endl;
        //     Vector2d y1_qr = Q1.transpose() * b1;
        //     Vector2d x1_qr = R1_qr.colPivHouseholderQr().solve(y1_qr);    // sto colPivHouseholder non mi piace molto


    }

    {

        Matrix2d A { {5.547001962252291e-01, -5.540607316466765e-01},
            { 8.320502943378437e-01,-8.324762492991313e-01}
        };

        Vector2d b {{-6.394645785530173e-04, 4.259549612877223e-04}};

        double condA,detA,errore_relativo_lu, errore_relativo_qr;

        if(SolveLUQRdecomposition(A,b,exactSolution, detA, condA, errore_relativo_lu, errore_relativo_qr)){
            cout<< "La Seconda matrice A : " <<endl<< A<<endl << " non e' singolare, con determinante "<<endl << detA<< endl << " e numero di condizionamento : " <<endl<< condA <<endl;
            cout<< "La soluzione del sistema con metodo di fattorizzazione LU, rispetto alla soluzione esatta "<<endl << exactSolution <<endl << "restituisce un errore relativo pari a "<<endl << errore_relativo_lu<<endl;
            cout << "La soluzione con fattorizzazione qr da un errore relativo pari a "<<endl << errore_relativo_qr << endl;
        }
        else{
            cout << "La Seconda matrice e' singolare" << endl;
        }

    }

    {
        Matrix2d A { {5.547001962252291e-01, -5.547001955851905e-01},
            {8.320502943378437e-01,-8.320502947645361e-01}
        };

        Vector2d b {{-6.400391328043042e-10, 4.266924591433963e-10}};

        double condA,detA,errore_relativo_lu, errore_relativo_qr;

        if(SolveLUQRdecomposition(A,b,exactSolution, detA, condA, errore_relativo_lu, errore_relativo_qr)){
            cout<< "La terza matrice A : " <<endl<< A<<endl << " non e' singolare, con determinante "<<endl << detA<< endl << " e numero di condizionamento : " <<endl<< condA <<endl;
            cout<< "La soluzione del sistema con metodo di fattorizzazione LU, rispetto alla soluzione esatta "<<endl << exactSolution <<endl << "restituisce un errore relativo pari a "<<endl << errore_relativo_lu<<endl;
            cout << "La soluzione con fattorizzazione qr da un errore relativo pari a "<<endl << errore_relativo_qr << endl;
        }
        else{
            cout << "La terza matrice e' singolare " << endl;
        }


    }
    return 0;
}
