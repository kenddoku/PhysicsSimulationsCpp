#include <iostream>
#include <vector>
#include <string>
#include <fstream>
#include <cmath>
#include <Eigen/Dense>

void arr2D_display(std::vector<std::vector<double>> &arr)
{
    for(int i=0; i<(int)arr.size(); i++)
    {
        for(int j=0; j<(int)arr[0].size(); j++)
        {
            std::cout << arr[i][j] << " ";
        }
        std::cout << "\n";
    }
}

double phi_k(double alpha_x, double alpha_y, double x_k, double y_k, double x, double y)
{
    double pi {std::numbers::pi};
    return (1./std::pow(alpha_x * pi, 1./4.)) * std::exp(-1 * (x-x_k)*(x-x_k) / (2*alpha_x)) *
            (1./std::pow(alpha_y * pi, 1./4.)) * std::exp(-1 * (y-y_k)*(y-y_k) / (2*alpha_y));
}

// *** Function filling matrices S, V, K, H
void fill_matrices(
            const int N, const double m, const double om_x, const double om_y, const double alpha_x,
            const double alpha_y, const std::vector<double>& x_k, const std::vector<double>& y_k,
            std::vector<std::vector<double>>& S, std::vector<std::vector<double>>& V,
            std::vector<std::vector<double>>& K, std::vector<std::vector<double>>& H
        )
{
    for(int k=0; k<N; ++k)
    {
        for(int l=0; l<N; ++l)
        {
            S[k][l] = std::exp(
                (-1)*(x_k[k] - x_k[l])*(x_k[k] - x_k[l]) -
                (-1)*(y_k[k] - y_k[l])*(y_k[k] - y_k[l])
            );

            V[k][l] = (1./2.)*m*(
                                    om_x*om_x * ((x_k[k]+x_k[l])*(x_k[k]+x_k[l]) + 2*alpha_x)/4. +
                                    om_y*om_y * ((y_k[k]+y_k[l])*(y_k[k]+y_k[l]) + 2*alpha_y)/4.
            )*S[k][l];

            K[k][l] = (-1.)/(2.*m)*(
                                    ((x_k[k]-x_k[l])*(x_k[k]-x_k[l]) - 2*alpha_x)/(4*alpha_x*alpha_x) +
                                    ((y_k[k]-y_k[l])*(y_k[k]-y_k[l]) - 2*alpha_y)/(4*alpha_y*alpha_y)
            )*S[k][l];

            H[k][l] = V[k][l] + K[k][l];
        }
    }
}

void fill_matrices2(
    const int N, const double m, const double om_x, const double om_y, const double alpha_x,
    const double alpha_y, const std::vector<double>& x_k, const std::vector<double>& y_k,
    Eigen::MatrixXd& S, Eigen::MatrixXd& V, Eigen::MatrixXd& K, Eigen::MatrixXd& H 
    )
{
    for(int k=0; k<N; ++k)
    {
        for(int l=0; l<N; ++l)
        {
            S(k, l) = std::exp(
                (-1)*(x_k[k] - x_k[l])*(x_k[k] - x_k[l]) / (4*alpha_x) +
                (-1)*(y_k[k] - y_k[l])*(y_k[k] - y_k[l]) / (4*alpha_y)
            );

            V(k, l) = (1./2.)*m*(
                                    om_x*om_x * ((x_k[k]+x_k[l])*(x_k[k]+x_k[l]) + 2*alpha_x)/4. +
                                    om_y*om_y * ((y_k[k]+y_k[l])*(y_k[k]+y_k[l]) + 2*alpha_y)/4.
            )*S(k, l);

            K(k, l) = (-1.)/(2.*m)*(
                                    ((x_k[k]-x_k[l])*(x_k[k]-x_k[l]) - 2*alpha_x)/(4*alpha_x*alpha_x) +
                                    ((y_k[k]-y_k[l])*(y_k[k]-y_k[l]) - 2*alpha_y)/(4*alpha_y*alpha_y)
            )*S(k, l);

            H(k, l) = V(k, l) + K(k, l);
        }
    }
}

// void calc_psi(Eigen::MatrixXd& psi, )
// {

// }

int main()
{
    // * Parametry siatki-------------------------------------------------------
    const int n {9};
    const int N {n*n};
    const double nm_to_at {0.0529177};
    double dx {1 / nm_to_at}; // delta_X [at]; {2 [nm] / ... [nm/at]}
    double a {dx * (double)(n-1)/2};

    // * Stałe -----------------------------------------------------------------
    const int h {1};
    const double E_h {27.211};  // przelicznik energii na jednostki atomowe [eV]
    const double m {0.24};      // masa efektywna elektronu

    // czestości omega_x oraz omega_y [ev / E_h]
    double om_x {0.08 / E_h};
    double om_y {0.2 / E_h};

    double alpha_x {h / (m * om_x)};
    double alpha_y {h / (m * om_y)};

    // Tworzenie siatki n x n
    std::vector<std::vector<double>> lattice(n, std::vector<double>(n, 0));

    // Tablice położeń węzłów
    std::vector<double> x_k(N, 0);
    std::vector<double> y_k(N, 0);

    for(int i=0; i<n; i++)
    {
        for(int j=0; j<n; j++)
        {
            int k {i * n + j};
            x_k[k] = (-1)*a + dx * i;
            y_k[k] = (-1)*a + dx * j;
        }
    }

    // Gęsta siatka do liczenia gaussjanów
    int n_dense {200};
    std::vector<std::vector<double>> dense_lattice(n_dense, std::vector<double>(n_dense, 0));

    std::vector<int> k_val {0, 8, 9};

    for(int indx=0; indx<(int)k_val.size(); indx++)
    {
        std::string filename = "zad1_k" + std::to_string(k_val[indx]) + ".csv";
        std::ofstream file(filename);
        if(!file.is_open()) std::cerr << "Error opening " << filename << "\n";

        std::vector<double> x_dense(n_dense*n_dense, 0);
        std::vector<double> y_dense(n_dense*n_dense, 0);
        double dx_dense = 2*a/n_dense;

        for(int i=0; i<n_dense; i++)
        {
            for(int j=0; j<n_dense; j++)
            {
                int k {i * n_dense + j};
                x_dense[k] = (-1)*a + dx_dense*i;
                y_dense[k] = (-1)*a + dx_dense*j;
                dense_lattice[i][j] = phi_k(alpha_x, alpha_y, x_k[k_val[indx]], y_k[k_val[indx]], x_dense[k], y_dense[k]);
                
                if(j == n_dense - 1) file << dense_lattice[i][j]; 
                else file << dense_lattice[i][j] << ", "; 
            }
            file << "\n";
        }

        file.close();
    }

    // *** MACIERZE S, V, K, H --------------------------------------------------
    // std::vector<std::vector<double>> S(N, std::vector<double>(N, 0));
    // std::vector<std::vector<double>> V(N, std::vector<double>(N, 0));
    // std::vector<std::vector<double>> K(N, std::vector<double>(N, 0));
    // std::vector<std::vector<double>> H(N, std::vector<double>(N, 0));

    // fill_matrices(N, m, om_x, om_y, alpha_x, alpha_y, x_k, y_k, S, V, K, H);

    // arr2D_display(S);

    Eigen::MatrixXd S(N, N);
    Eigen::MatrixXd V(N, N);
    Eigen::MatrixXd K(N, N);
    Eigen::MatrixXd H(N, N);

    fill_matrices2(N, m, om_x, om_y, alpha_x, alpha_y, x_k, y_k, S, V, K, H);

    Eigen::GeneralizedSelfAdjointEigenSolver<Eigen::MatrixXd> solver(H, S);
    Eigen::VectorXd eigenValues = solver.eigenvalues();
    Eigen::MatrixXd eigenVectors = solver.eigenvectors();

    // if(solver.info() == Eigen::Success)
    // {
    //     std::cout << "Eigenvalues:\n" << solver.eigenvalues() << std::endl;
    //     std::cout << "Eigenvectors:\n" << solver.eigenvectors() << std::endl;
    // }
    // else std::cout << "Something went wrong with the solver" << std::endl;

    Eigen::MatrixXd PSI(n_dense, n_dense);
    PSI.setZero();

    std::vector<double> x_dense(n_dense*n_dense, 0);
    std::vector<double> y_dense(n_dense*n_dense, 0);
    double dx_dense = 2*a/n_dense;

    for(int i=0; i<n_dense; i++)
    {
        for(int j=0; j<n_dense; j++)
        {
            int k {i * n_dense + j};
            x_dense[k] = (-1)*a + dx_dense*i;
            y_dense[k] = (-1)*a + dx_dense*j;
        }
    }

    for(int e=0; e<6; ++e)
    {
        std::string filename = "psiE" + std::to_string(e) + ".csv";
        std::ofstream file_psi;
        file_psi.open(filename);
        for(int i=0; i<n_dense; ++i)
        {
            for(int j=0; j<n_dense; ++j)
            {
                int k {i*n_dense + j};

                for(int nod=0; nod<N; ++nod)
                {
                    PSI(i, j) += eigenVectors(nod, e) * phi_k(alpha_x, alpha_y, x_k[nod], y_k[nod], x_dense[k], y_dense[k]);
                }

                PSI(i, j) = PSI(i, j) * PSI(i, j);
            }
        }
        file_psi << PSI;
        file_psi.close();
    }


    // *** Changing om_x values and chechick how it affects eigenvalues for 10 first states 
    std::vector<double> om_x_val {};
    double om_x_max {0.500 / E_h};
    int om_x_points {500};
    double delta_om_x {om_x_max / om_x_points};

    for(int i=0; i<om_x_points; ++i)
    {
        om_x_val.push_back((i+20)*delta_om_x); // (i+1) so it does not start from zero
    }

    std::string filename = "omx_E.csv";
    std::ofstream file;
    file.open(filename);
    if(!file.is_open()) std::cerr << "ERROR opening " << filename << std::endl;

    for(size_t i=0; i<om_x_val.size(); ++i)
    {
        alpha_x = h/(m*om_x_val[i]);

        S.setZero();
        V.setZero();
        K.setZero();
        H.setZero();

        fill_matrices2(N, m, om_x_val[i], om_y, alpha_x, alpha_y, x_k, y_k, S, V, K, H);

        Eigen::GeneralizedSelfAdjointEigenSolver<Eigen::MatrixXd> solver(H, S);
        Eigen::VectorXd eigenValues = solver.eigenvalues();
        Eigen::MatrixXd eigenVectors = solver.eigenvectors();

        file << om_x_val[i] * E_h << " " << eigenValues.transpose() << "\n";
    }
    //dense_lattice[i][j] = phi_k(alpha_x, alpha_y, x_k[k_val[indx]], y_k[k_val[indx]], x_dense[k], y_dense[k]);
    file.close();

    return 0;
}