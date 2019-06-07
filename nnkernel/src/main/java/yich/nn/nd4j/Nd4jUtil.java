package yich.nn.nd4j;

import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.TransformOp;
import org.nd4j.linalg.api.ops.impl.transforms.Exp;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.ops.transforms.Transforms;

public class Nd4jUtil {

    // apply a function to every element in a array
    public static INDArray apply(INDArray arr, NumFunc func) {
        long len = arr.length();
//        Transforms.exp(arr, false);
//        TransformOp exp = new Exp();
//        Nd4j.getExecutioner().execAndReturn(exp);
        // arr.exp
        for (int i = 0; i < len; i++) {
            arr.putScalar(i, func.apply(arr.getDouble(i)));
        }
        return arr;
    }

}
