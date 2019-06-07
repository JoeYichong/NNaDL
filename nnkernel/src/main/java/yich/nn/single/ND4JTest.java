package yich.nn.single;

import org.nd4j.linalg.api.blas.params.MMulTranspose;
import org.nd4j.linalg.api.buffer.DataBuffer;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.rng.distribution.Distribution;
import org.nd4j.linalg.api.rng.distribution.impl.NormalDistribution;
import org.nd4j.linalg.factory.Nd4j;

public class ND4JTest {

    public static void main(String[] args) {
//        INDArray array2 = Nd4j.zeros(2,3);
//        INDArray oneDArray = Nd4j.create(new float[]{1,2,3,4,5,6} , new int[]{6});
//
//        System.out.println(array2);
//        System.out.println(oneDArray);

        // Nd4j.randn()
        INDArray a1 = Nd4j.rand(new int[]{2,3}, new NormalDistribution());
        INDArray a2 = Nd4j.rand(new int[]{3,2}, new NormalDistribution());
        INDArray a3 = Nd4j.rand(new int[]{2,3}, new NormalDistribution());
        INDArray a4 = Nd4j.rand(new int[]{3,3}, new NormalDistribution());
        INDArray a5 = Nd4j.create(new int[]{2, 3});
        INDArray a6;
        INDArray a7 = Nd4j.create(new int[]{2, 2});;

//        System.out.println(a1.mmul(a3, MMulTranspose.builder().build()));
        System.out.println(a1.mmul(a2, a7));
        System.out.println("a1: \n" + a1);
        System.out.println("a2: \n" + a2);
        System.out.println("a7: \n" + a7);


//        Nd4j.clearNans(arr.divi(0));
//        System.out.println("a1: \n" + a1);

//        System.out.println("a3: \n" + a3);
//        System.out.println("a5: \n" + a5);
//        System.out.println("a3.mmuli(a4, a5): \n" + (a6 = a3.mmul(a4, a5)));
//        System.out.println("a3: \n" + a3);
//        System.out.println("a5: \n" + a5);
//        System.out.println(a6 == a5) ;

//        System.out.println(a1.mmul(a3));



    }

}
