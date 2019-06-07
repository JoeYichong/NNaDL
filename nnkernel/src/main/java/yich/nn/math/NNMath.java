package yich.nn.math;

import org.nd4j.linalg.api.buffer.DataBuffer;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import yich.nn.nd4j.Nd4jUtil;

public class NNMath {
    public static double sigmoid(double z) {
        return 1 / (1 + Math.exp(-z));
    }

    public static INDArray sigmoid(INDArray z) {
        // dz is a copy
//        double[] dz = z.data().asDouble();
//        for (int i = 0; i < dz.length; i++) {
//            dz[i] = sigmoid(dz[i]);
//        }
//        return Nd4j.create(dz, z.shape());

        return Nd4jUtil.apply(z, NNMath::sigmoid);
    }


    public static double sigmoid_prime(double z) {
        return sigmoid(z) * (1 - sigmoid(z));
    }

    public static INDArray sigmoid_prime(INDArray z) {
//        DataBuffer db = z.data();
        double[] dz = z.data().asDouble();
        for (int i = 0; i < dz.length; i++) {
            dz[i] = sigmoid_prime(dz[i]);
        }
        return Nd4j.create(dz, z.shape());
//        db.setData(dz);
//        z.setData(db);
//        return z;

//        return Nd4jUtil.apply(z, NNMath::sigmoid_prime);
    }


    public static double cross_entropy(double a, double y) {
        if ( y == 0) {
            return -Math.log1p(-a);
        } else if (y == 1) {
            return -Math.log(a);
        } else {
            return -(y * Math.log(a) + (1 - y) * Math.log1p(-a)); // log1p(-a) is equivalent to log(1-a)
        }

    }

    public static double cross_entropy(INDArray a, INDArray y) {
        double[] da = a.data().asDouble();
        double[] dy = a.data().asDouble();
        for (int i = 0; i < da.length; i++) {
            dy[i] = cross_entropy(da[i], dy[i]);
        }
        return Nd4j.create(dy, y.shape()).sumNumber().doubleValue() / (double) y.length();

    }

    public static double quadratic_cost(double a, double y) {
        double d = a - y;
        return d * d * 0.5;
//        return Math.pow(a - y, 2) * 0.5;
    }



//    public static void main(String[] args) {
//
//        System.out.println(cross_entropy(0, 0));
//        System.out.println(cross_entropy(0.1, 0.1));
//        System.out.println(cross_entropy(0.2, 0.2));
//        System.out.println(cross_entropy(0.3, 0.3));
//        System.out.println(cross_entropy(0.4, 0.4));
//        System.out.println(cross_entropy(0.5, 0.5));
//        System.out.println(cross_entropy(0.6, 0.6));
//        System.out.println(cross_entropy(0.7, 0.7));
//        System.out.println(cross_entropy(0.8, 0.8));
//        System.out.println(cross_entropy(0.9, 0.9));
//        System.out.println(cross_entropy(1, 1));
//    }

}
