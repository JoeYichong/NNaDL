package yich.nn.single;

import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.rng.distribution.impl.NormalDistribution;
import org.nd4j.linalg.factory.Nd4j;
import yich.nn.loader.DataPath;
import yich.nn.loader.IDXFileParser;
import yich.nn.loader.IDXObject;

public class NeuralNetTest {
    INDArray weights = Nd4j.rand(new int[]{30, 784}, new NormalDistribution(0, 1));
    INDArray biases = Nd4j.rand(new int[]{30, 1}, new NormalDistribution(0, 1));
    IDXObject images = IDXFileParser.parse(DataPath.MNIST.getProperty("test.images"));
    IDXObject labels = IDXFileParser.parse(DataPath.MNIST.getProperty("test.labels"));
    int index = 1559;

    public INDArray feedforward(INDArray x) {
        return weights.mmul(x).add(biases);
    }

    public void run() {
//        INDArray x = Nd4j.create(images.getAsDouble(index), new int[]{784, 1});
//        INDArray r = feedforward(x.div(255));
//        System.out.println(r);

//        INDArray v1 = Nd4j.create(new double[]{1, 2, 3}, new int[]{3, 1});
//        INDArray v2 = Nd4j.create(new double[]{4, 3, 2, 1}, new int[]{1, 4});
//        System.out.println(v1.broadcast(3, 4).mul(v2.broadcast(3, 4)));

//        INDArray x = Nd4j.create(new double[]{1, 2, 3, 4, 5, 6}, new int[]{2, 3});
//        System.out.println(x.length());
//        System.out.println(x.getDouble(4));

    }

    public static void main(String[] args) {
        new NeuralNetTest().run();
//        System.out.println(t.weights);
//        System.out.println(Arrays.toString(t.images.getAsDouble(index)));
        // t.images.printInfo();
        // System.out.println(t.labels.getData()[index]);
        // MNISTParser.printImage(t.images, index);

    }

}
