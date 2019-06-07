package yich.nn.single;

import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.rng.distribution.impl.NormalDistribution;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.ops.transforms.Transforms;
import yich.nn.loader.DataPath;
import yich.nn.loader.IDXFileParser;
import yich.nn.loader.IDXObject;
import yich.nn.plot.NNChart;
import yich.nn.plot.Plotter;
import yich.nn.math.NNMath;

public class NeuralNet {
    int[] sizes;
    INDArray[] weights;
    INDArray[] biases;

    int epoch = 50;
    double eta = 0.5;
    double lambda = 3;

    IDXObject images = IDXFileParser.parse(DataPath.MNIST.getProperty("test.images"));
    IDXObject labels = IDXFileParser.parse(DataPath.MNIST.getProperty("test.labels"));
    int index = 2559;

    INDArray y;

    INDArray[] zs; // z
    INDArray[] as;     // a
    INDArray[] deltas;
    INDArray[] nabla_ws;
    INDArray[] nabla_bs;
    double[] costs;


    public NeuralNet(int[] sizes) {
        this.sizes = sizes;
        this.weights = new INDArray[sizes.length - 1];
        this.biases = new INDArray[sizes.length - 1];

        for (int i = 0; i < weights.length; i++) {
            weights[i] = Nd4j.rand(new int[]{sizes[i + 1], sizes[i]},
                                    new NormalDistribution(0, Math.sqrt((double) 1 / (double) sizes[i])));
        }

        for (int i = 0; i < biases.length; i++) {
            biases[i] = Nd4j.rand(new int[]{sizes[i + 1], 1}, new NormalDistribution(0, 1));
        }

        //
        y = convert(labels.getData()[index]);
        //
        zs = new INDArray[biases.length]; // z
        for (int i = 0; i < zs.length; i++) {
            zs[i] = Nd4j.create(new int[]{sizes[i + 1], 1});
        }


        as = new INDArray[sizes.length];     // a
        for (int i = 0; i < as.length; i++) {
            as[i] = Nd4j.create(new int[]{sizes[i], 1});
        }


        deltas = new INDArray[zs.length];
        for (int i = 0; i < deltas.length; i++) {
            deltas[i] = Nd4j.create(new int[]{sizes[i + 1], 1});
        }


        nabla_ws = new INDArray[weights.length];
        for (int i = 0; i < nabla_ws.length; i++) {
//            nabla_ws[i] = Nd4j.create(new int[]{sizes[i + 1], sizes[i]});
            nabla_ws[i] = Nd4j.zeros(weights[i].shape());
        }

        nabla_bs = new INDArray[biases.length];

        costs = new double[epoch];
    }

    public INDArray activate(INDArray z) {
        return NNMath.sigmoid(z);
//        return Transforms.sigmoid(z);
    }

    // compute 'z' & 'a'
    public void feedforward(INDArray x) {
        as[0] = x;
        for (int i = 0; i < weights.length; i++) {
            weights[i].mmuli(as[i], zs[i]).addi(biases[i]); // Note that 'mmuli' and 'mmul' are the same here
            as[i + 1] = activate(zs[i]);
        }
    }

    // compute 'delta'
    public void compute_delta() {
        int i = deltas.length - 1;
        as[as.length - 1].subi(y, deltas[i--]); // a:(m*1) - y:(m*1) = delta:(m*1)
        for (; i >= 0; i--) {
            deltas[i + 1].transpose()                            // delta:(m*1) => delta:(1*m)
                          .mmul(weights[i + 1]).transposei()     // weights:(m*n) | delta:(1*n) => delta:(n*1)
//                         .mul(Transforms.sigmoidDerivative(zs[i]));
                          .mul(NNMath.sigmoid_prime(zs[i]), deltas[i]);     // zs:(n*1)
        }
    }

    public void compute_updatingMatrices() {

        for (int i = deltas.length - 1; i >= 0; i--) {
//            nabla_ws[i] = deltas[i].broadcast(deltas[i].length(), as[i].length())
//                                   .mul(as[i].transpose().broadcast(deltas[i].length(), as[i].length()));
//            nabla_bs[i] = deltas[i].dup();

//            nabla_ws[i] = deltas[i].broadcast(sizes[i + 1], sizes[i])
//                    .mul(as[i].transpose().broadcast(sizes[i + 1], sizes[i]));
//            nabla_bs[i] = deltas[i];

//            nabla_ws[i] = Nd4j.create(new int[]{sizes[i + 1], sizes[i]});
            deltas[i].broadcast(nabla_ws[i])
                     .muli(as[i].transpose().broadcast(nabla_ws[i].shape()));
            nabla_bs[i] = deltas[i];

        }
    }

    public void update() {
//        for (int i = deltas.length - 1; i >= 0; i--) {
//            deltas[i].broadcast(nabla_ws[i])
//                    .muli(as[i].transpose().broadcast(nabla_ws[i].shape()));
//            nabla_bs[i] = deltas[i];
//            weights[i].subi(nabla_ws[i].muli(eta));
//            biases[i].subi(nabla_bs[i].muli(eta));
//        }
        for (int i = 0; i < weights.length; i++){
            weights[i].subi(nabla_ws[i].muli(eta));
            biases[i].subi(nabla_bs[i].muli(eta));
        }
    }


    public double cost(INDArray a) {
        return NNMath.cross_entropy(a, y);
    }

    // convert a number to a corresponding array
    private INDArray convert(int num) {
        int yn = sizes[sizes.length - 1];
        double[] arr = new double[yn];
        arr[num % yn] = 1;
        return Nd4j.create(arr, new int[]{yn, 1});
    }


    public void run() {

        // extract an image example and feed it to the net
        INDArray x = Nd4j.create(images.getAsDouble(index), new int[]{784, 1});
        x.divi(255);
        System.out.println("y=" + labels.getData()[index]);

//        feedforward(x);
//        compute_delta();
//        compute_updatingMatrices();

//        for (int i = 0; i < deltas.length; i++) {
//            System.out.println(i + ": ");
//            System.out.println(deltas[i]);
//        }

//        for (int i = 0; i < nabla_ws.length; i++) {
//            System.out.println(i + ": ");
//            System.out.println("Shape: " + Arrays.toString(nabla_ws[i].shape()));
//            System.out.println(nabla_ws[i].sumNumber());
//            System.out.println(nabla_ws[i]);
//        }

//        for (int i = 0; i < nabla_bs.length; i++) {
//            System.out.println(i + ": ");
//            System.out.println("Shape: " + Arrays.toString(nabla_bs[i].shape()));
//            System.out.println(nabla_bs[i].sumNumber());
//            System.out.println(nabla_bs[i]);
//        }

        long startTime = System.nanoTime();

        for (int i = 0; i < epoch; i++) {
            feedforward(x);
            compute_delta();
            compute_updatingMatrices();
            update();
            costs[i] = cost(as[as.length - 1]);
        }
        System.out.println(as[as.length - 1]);
        System.out.println(y);

        long endTime   = System.nanoTime();
        long totalTime = endTime - startTime;
        double seconds = (double) totalTime / 1_000_000_000.0;
        System.out.println("Total Time: " + seconds + " seconds");


//        Plotter plotter = new NNChart();
//        plotter.plot(costs);


    }

    public static void main(String[] args) {
        new NeuralNet(new int[]{784, 30 , 10}).run();

    }

}
