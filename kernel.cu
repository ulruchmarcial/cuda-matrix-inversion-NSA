
#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <math.h>

#define N 64
#define TILE_WIDTH 16

// Macro de gestion d'erreur
#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            fprintf(stderr, "CUDA Error: %s (file %s, line %d)\n", \
                    cudaGetErrorString(err), __FILE__, __LINE__); \
            exit(EXIT_FAILURE); \
        } \
    } while(0)

// ======================= KERNELS CUDA ======================= //

// Multiplication matricielle C = A * B avec tiling (64x64, tiles 16x16)
__global__ void matMulTiled(const float* A, const float* B, float* C, int n) {
    __shared__ float As[TILE_WIDTH][TILE_WIDTH];
    __shared__ float Bs[TILE_WIDTH][TILE_WIDTH];

    int tx = threadIdx.x;
    int ty = threadIdx.y;

    int row = blockIdx.y * TILE_WIDTH + ty;
    int col = blockIdx.x * TILE_WIDTH + tx;

    float sum = 0.0f;
    int numTiles = (n + TILE_WIDTH - 1) / TILE_WIDTH;

    for (int t = 0; t < numTiles; ++t) {
        int tiledColA = t * TILE_WIDTH + tx;
        int tiledRowB = t * TILE_WIDTH + ty;

        // Charger A dans As
        if (row < n && tiledColA < n)
            As[ty][tx] = A[row * n + tiledColA];
        else
            As[ty][tx] = 0.0f;

        // Charger B dans Bs
        if (tiledRowB < n && col < n)
            Bs[ty][tx] = B[tiledRowB * n + col];
        else
            Bs[ty][tx] = 0.0f;

        __syncthreads();

        // Produit partiel
#pragma unroll
        for (int k = 0; k < TILE_WIDTH; ++k) {
            sum += As[ty][k] * Bs[k][tx];
        }

        __syncthreads();
    }

    if (row < n && col < n)
        C[row * n + col] = sum;
}

// Scaling à gauche : B = Dinv * E
// Dinv est donnée sous forme de vecteur diag : dInvDiag[row]
__global__ void left_scale(const float* E, float* B, const float* dInvDiag, int n) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < n && col < n) {
        int idx = row * n + col;
        B[idx] = dInvDiag[row] * E[idx];
    }
}

// Scaling à droite : Y = X * Dinv
// Dinv est donnée sous forme de vecteur diag : dInvDiag[col]
__global__ void right_scale(const float* X, float* Y, const float* dInvDiag, int n) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < n && col < n) {
        int idx = row * n + col;
        Y[idx] = X[idx] * dInvDiag[col];
    }
}

// Combinaison finale : Ainv = Dinv - M2 + Term3
// Dinv est diagonale → seulement sur row == col
__global__ void combine_result(float* Ainv,
    const float* dInvDiag,
    const float* M2,
    const float* Term3,
    int n) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < n && col < n) {
        int idx = row * n + col;
        float dterm = (row == col) ? dInvDiag[row] : 0.0f;
        Ainv[idx] = dterm - M2[idx] + Term3[idx];
    }
}

// =================== LOGIQUE NSA ORDRE 3 =================== //
// Approximation : A^{-1} ≈ D^{-1} - D^{-1} E D^{-1} + (D^{-1}E)^2 D^{-1}
//
// Entrées GPU :
//  - dInvDiag : vecteur (n) contenant 1 / A[i,i]
//  - d_E      : matrice n x n (off-diagonale de A)
// Sortie GPU :
//  - d_result : approximation de A^{-1}
void neumannOrder3(const float* dInvDiag, const float* d_E, float* d_result, int n) {
    size_t sizeMat = n * n * sizeof(float);
    float* d_M1, * d_M2, * d_M3, * d_Term3;

    CUDA_CHECK(cudaMalloc(&d_M1, sizeMat));
    CUDA_CHECK(cudaMalloc(&d_M2, sizeMat));
    CUDA_CHECK(cudaMalloc(&d_M3, sizeMat));
    CUDA_CHECK(cudaMalloc(&d_Term3, sizeMat));

    dim3 block(TILE_WIDTH, TILE_WIDTH);
    dim3 grid((n + TILE_WIDTH - 1) / TILE_WIDTH,
        (n + TILE_WIDTH - 1) / TILE_WIDTH);

    // 1) M1 = D^{-1} * E  (scaling des lignes)
    left_scale << <grid, block >> > (d_E, d_M1, dInvDiag, n);

    // 2) M2 = M1 * D^{-1}  (scaling des colonnes)
    right_scale << <grid, block >> > (d_M1, d_M2, dInvDiag, n);

    // 3) M3 = M1 * M1  = (D^{-1} E)^2    (seule vraie matmul tiled)
    matMulTiled << <grid, block >> > (d_M1, d_M1, d_M3, n);

    // 4) Term3 = M3 * D^{-1}  (scaling des colonnes)
    right_scale << <grid, block >> > (d_M3, d_Term3, dInvDiag, n);

    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    // 5) Ainv = D^{-1} - M2 + Term3
    combine_result << <grid, block >> > (d_result, dInvDiag, d_M2, d_Term3, n);

    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    cudaFree(d_M1);
    cudaFree(d_M2);
    cudaFree(d_M3);
    cudaFree(d_Term3);
}


// fonction pour la charger le fichier matrixA.csv contenant la matrice A  //

void loadMatrixCSV(const char* filename, float A[N][N]) {
    FILE* f = fopen(filename, "r");
    if (!f) {
        printf("Fichier %s introuvable. Utilisation de données aléatoires.\n", filename);
        // generateMatrix(A);
        return;
    }
    for (int i = 0; i < N; i++)
        for (int j = 0; j < N; j++)
            fscanf(f, "%f,", &A[i][j]);
    fclose(f);
}

// calcule de l’erreur de Frobenius pour vérifier si la matrice inverse du GPU == matrice inverse de mathlab.

float frobeniusNorm(const float M[N][N]) {
    float sum = 0.0f;
    for (int i = 0; i < N; i++)
        for (int j = 0; j < N; j++)
            sum += M[i][j] * M[i][j];
    return sqrtf(sum);
}

// =============================== MAIN =============================== //

int main() {
    float A[N][N];
    float D[N][N] = { 0 };
    float E[N][N] = { 0 };
    float dInvDiag_h[N];        // diagonale de D^{-1}
    float Ainv_app[N][N];
    float Ainv_exact[N][N];

    // 1)  Chargement de la matrice A et ma matrice inverse de A provenant de Matlab
    loadMatrixCSV("C:\\Users\\ngul4685\\Desktop\\Nouveau dossier\\matrixA.csv", A);
    loadMatrixCSV("C:\\Users\\ngul4685\\Desktop\\Nouveau dossier\\matrix_inv_A.csv", Ainv_exact);

    // 2) Construction de D, E et D^{-1} (diag)
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            if (i == j) {
                D[i][j] = A[i][j];
                dInvDiag_h[i] = 1.0f / A[i][j];
                E[i][j] = 0.0f;
            }
            else {
                D[i][j] = 0.0f;
                E[i][j] = A[i][j];
            }
        }
    }

    // 3) Allocation de la mémoire sur le GPU 
    size_t sizeMat = N * N * sizeof(float);
    float* d_E, * d_result;
    float* d_dInvDiag;

    CUDA_CHECK(cudaMalloc(&d_E, sizeMat));
    CUDA_CHECK(cudaMalloc(&d_result, sizeMat));
    CUDA_CHECK(cudaMalloc(&d_dInvDiag, N * sizeof(float)));

    CUDA_CHECK(cudaMemcpy(d_E, E, sizeMat, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_dInvDiag, dInvDiag_h, N * sizeof(float), cudaMemcpyHostToDevice));

    // 4) Approximation NSA ordre 3
    neumannOrder3(d_dInvDiag, d_E, d_result, N);

    // 5) Récupération du résultat
    CUDA_CHECK(cudaMemcpy(Ainv_app, d_result, sizeMat, cudaMemcpyDeviceToHost));


    // Calculer la différence : Diff = Ainv_exact - Ainv_app
    // il s'agit de la difference  entre la matrice inverse A provenant de matlab et celle inverseé par CUDA c 
    float Diff[N][N];
    for (int i = 0; i < N; i++)
        for (int j = 0; j < N; j++)
            Diff[i][j] = Ainv_exact[i][j] - Ainv_app[i][j];

    float normExact = frobeniusNorm(Ainv_exact);
    float normDiff = frobeniusNorm(Diff);

    printf("\nErreur relative Frobenius ||Ainv_exact - Ainv_app|| / ||Ainv_exact|| = %e\n",
        normDiff / normExact);

    // --- Affichages partiels pour vérifier --- //
    printf("\nMatrice A (64 x 64):\n");
    for (int i = 0; i < 10; i++) {
        for (int j = 0; j < 10; j++)
            printf("%8.4f ", A[i][j]);
        printf("\n");
    }

    printf("\nMatrice D (64 x 64):\n");
    for (int i = 0; i < 10; i++) {
        for (int j = 0; j < 10; j++)
            printf("%8.4f ", D[i][j]);
        printf("\n");
    }

    printf("\nDiag(Dinv) (64 x 64) :\n");
    for (int i = 0; i < 10; i++) {
        for (int j = 0; j < 10; j++) {
            if (i == j)
                printf("%8.4f ", dInvDiag_h[i]);
            else
                printf("%8.4f ", 0.0f);
        }
        printf("\n");
    }


    printf("\nMatrice E (64 x 64):\n");
    for (int i = 0; i < 10; i++) {
        for (int j = 0; j < 10; j++)
            printf("%8.4f ", E[i][j]);
        printf("\n");
    }

    printf("\nApprox A^{-1} (64 x 64):\n");
    for (int i = 0; i < 10; i++) {
        for (int j = 0; j < 10; j++)
            printf("%8.4f ", Ainv_app[i][j]);
        printf("\n");
    }

    // Vérification rapide : A * Ainv ≈ I 
    printf("\nVérification (A * Ainv approx) - 64x64 :\n");
    for (int i = 0; i < 64; i++) {
        for (int j = 0; j < 64; j++) {
            float sum = 0.0f;
            for (int k = 0; k < N; k++)
                sum += A[i][k] * Ainv_app[k][j];
            printf("%6.2f ", sum);
        }
        printf("\n");
    }
    // --- liberation de la memoire GPU --- //
    cudaFree(d_E);
    cudaFree(d_result);
    cudaFree(d_dInvDiag);

    return 0;
}
