#include <stdio.h>
#include <stdlib.h>
#include <math.h>

typedef struct{
    double *data;
    int rows;
    int cols;
    int size;
    char *name;
}Tensor;

Tensor create_tensor(int rows, int cols, char *name)
{
    Tensor tensor;
    tensor.rows = rows;
    tensor.cols = cols;
    tensor.size = rows * cols;
    tensor.name = name;
    tensor.data = (double *)malloc(tensor.size * sizeof(double));
    return tensor;
}

void init_tensor(Tensor tensor)
{
    for (int i = 0; i < tensor.size; i++)
        tensor.data[i] = 2*((double)rand() / RAND_MAX) - 1;
}

void init_img(Tensor tensor)
{
    for (int i = 0; i < tensor.size; i++)
        tensor.data[i] = (double)rand() / RAND_MAX;
}

void init_label(Tensor tensor)
{
    for (int i = 0; i < tensor.size; i++)
        tensor.data[i] = rand()%2;
}

void free_tensor(Tensor tensor)
{
    free(tensor.data);
}

void update_tensor(Tensor tensor, double error)
{
    //double update = 2*error - 1;
    double update = error;
    //printf("Error: %f, Update: %f\n", error, update);
    for (int i = 0; i < tensor.size; i++){
        tensor.data[i] += update; 
        tensor.data[i] /= 2;
        if (tensor.data[i] < -1)
            tensor.data[i] += 1;
        if (tensor.data[i] > 1)
            tensor.data[i] -= 1;    
    }
}

Tensor operate_tensor(Tensor tensor1, Tensor tensor2, Tensor result, double bias, Tensor *gate)
{   
    int m = tensor1.rows;
    int n = tensor1.cols;
    int p = tensor2.cols;
    
    for (int i = 0; i < m; i++)
    {
        double tMax, tMin;
        result.data[i * p] = bias;
        for (int k = 0; k < n; k++)
            result.data[i * p] += tensor1.data[i * n + k] * tensor2.data[k * n];
            
        if (gate != NULL)
            result.data[i * p ] += gate->data[i * p];
        tMax = result.data[i * p];
        tMin = result.data[i * p];
        for (int j = 1; j < p; j++){
            result.data[i * p + j] = bias;
            for (int k = 0; k < n; k++)
                result.data[i * p + j] += tensor1.data[i * n + k] * tensor2.data[k * n + j];
            if (gate !=NULL)
                result.data[i * p + j] += gate->data[i * p + j];
            if (result.data[i * p + j] < tMin)
                tMin = result.data[i * p + j];
            else if (result.data[i * p + j] > tMax)
                tMax = result.data[i * p + j];
        }
        for (int j = 0; j < p; j++)
            result.data[i * p + j] = 2*(result.data[i * p + j] - tMin) / (tMax - tMin) - 1;
    }
    return result;
}

void sigmoid(Tensor tensor)
{
    tensor.name = "sigmoid";
    double inMin, inMax, outMin, outMax;
    inMin = tensor.data[0];
    inMax = tensor.data[0];
    tensor.data[0] = 1.0 / (1.0 + exp(-tensor.data[0]));
    outMin = tensor.data[0];
    outMax = tensor.data[0];
     
    for (int i = 1; i < tensor.size; i++){
        if (tensor.data[i] < inMin)
            inMin = tensor.data[i];
        else if (tensor.data[i] > inMax)
            inMax = tensor.data[i];
        tensor.data[i] = 1.0 / (1.0 + exp(-tensor.data[i]));
        if (tensor.data[i] < outMin)
            outMin = tensor.data[i];
        else if (tensor.data[i] > outMax)
            outMax = tensor.data[i];
    }
    printf("Sigmoid: inMin = %f, inMax = %f, outMin = %f, outMax = %f\n", inMin, inMax, outMin, outMax);
}

void limiar(Tensor tensor)
{
    tensor.name = "limiar";
    for (int i = 0; i < tensor.size; i++)
        tensor.data[i] = (tensor.data[i] > 0) ? 1 : 0;
}

void print_tensor(Tensor tensor)
{
    printf("%s: ", tensor.name);
    printf("rows = %d, cols = %d, size = %d\n", tensor.rows, tensor.cols, tensor.size);
    printf("______________________________\n");
    int m,n;
    m = (tensor.rows > 16) ? 16 : tensor.rows;
    n = (tensor.cols > 16) ? 16 : tensor.cols;
    for (int i = 0; i < m; i++){
        for (int j = 0; j < n; j++)
            printf("%.4f ", tensor.data[i * tensor.cols + j]);
        printf("\n");
    }
}

double error(Tensor tensor1, Tensor tensor2)
{
    double sum = 0;
    double signal = 0;
    for (int i = 0; i < tensor1.size; i++)
    {
        sum += (tensor1.data[i] - tensor2.data[i]) * (tensor1.data[i] - tensor2.data[i]);
        signal += (tensor1.data[i] - tensor2.data[i]);
    }
    signal = (signal > 0) ? -1.0 : 1.0;
    sum = sum / tensor1.size;
    sum = sqrt(sum);
    return signal*sum;
}


int main()
{
    int m, n, p, k, l, batchSize;
    m = 1024;
    n = 1024;
    p = 512;
    k = 256;
    l = 128;
    batchSize = 8;
    Tensor img, ker_np, ker_pk, ker_kl, ker_lk, ker_kp, ker_pn, out, label;
    Tensor out_1, out_2, out_3, out_4, out_5;
    img = create_tensor(m, n, "img");
    init_img(img);
    //print_tensor(img);
    label = create_tensor(m, n, "label");
    init_label(label);
    //print_tensor(label);
    
    
    ker_np = create_tensor(n, p, "ker_np");
    init_tensor(ker_np);
    //print_tensor(ker_np);
    ker_pk = create_tensor(p, k, "ker_pk");
    init_tensor(ker_pk);
    //print_tensor(ker_pk);
    ker_kl = create_tensor(k, l, "ker_kl");
    init_tensor(ker_kl);
    //print_tensor(ker_kl);
    ker_lk = create_tensor(l, k, "ker_lk");
    init_tensor(ker_lk);
    //print_tensor(ker_lk);
    ker_kp = create_tensor(k, p, "ker_kp");
    init_tensor(ker_kp);
    //print_tensor(ker_kp);
    ker_pn = create_tensor(p, n, "ker_pn");
    init_tensor(ker_pn);
    //print_tensor(ker_pn);
    
    
    int epochs = 0;
    out_1 = create_tensor(batchSize, p, "out_1");
    out_2 = create_tensor(batchSize, k, "out_2");
    out_3 = create_tensor(batchSize, l, "out_3");
    out_4 = create_tensor(batchSize, k, "out_4");
    out_5 = create_tensor(batchSize, p, "out_5");
    out = create_tensor(batchSize, n, "out");

    Tensor batchImg;
    batchImg.rows = batchSize;
    batchImg.cols = n;
    batchImg.size = batchSize * n;
    batchImg.name = "batch";
    Tensor batchLbl;
    batchLbl.rows = batchSize;
    batchLbl.cols = n;
    batchLbl.size = batchSize * n;
    batchLbl.name = "batch label";

    while(epochs < 100)
    {
        int ini = 0;
        double errEpoch = 0;
        while(ini < m)
        {
            batchImg.data = img.data + ini * n;
            //print_tensor(batchImg);
            operate_tensor(batchImg, ker_np, out_1, 0.01, NULL);     //mxn * nxp -> mxp
            //print_tensor(out_1);
            operate_tensor(out_1, ker_pk, out_2, 0.02, NULL);   //mxp * pxk -> mxk
            //print_tensor(out_2);
            operate_tensor(out_2, ker_kl, out_3, 0.03, NULL);   //mxk * kxl -> mxl
            //print_tensor(out_3);
            operate_tensor(out_3, ker_lk, out_4, 0.03, &out_2); //mxl * lxm -> mxk
            //print_tensor(out_4);
            operate_tensor(out_4, ker_kp, out_5, 0.02, &out_1); //mxk * kxp -> mxp
            //print_tensor(out_5);
            operate_tensor(out_5, ker_pn, out, 0.01, NULL);     //mxp * pxn -> mxn
            //print_tensor(out);
            //sigmoid(out);
            limiar(out);
            //print_tensor(out);
            batchLbl.data = label.data + ini * n;
            double err = error(out, batchLbl);
            //printf("Error: %f\n", err);
            errEpoch += fabs(err);
            update_tensor(ker_kl, err);
            update_tensor(ker_lk, -err);
            update_tensor(ker_kp, err);
            update_tensor(ker_np, -err);
            update_tensor(ker_pk, err);
            update_tensor(ker_pn, -err);
            ini += batchSize;
        }
        epochs++;
        errEpoch = errEpoch / (m/batchSize);
        printf("Epoch %d: Error = %f\n", epochs, errEpoch);
        
    }
        /**/
    return 0;

}