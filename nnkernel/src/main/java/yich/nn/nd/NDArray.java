package yich.nn.nd;

import org.nd4j.linalg.api.ndarray.INDArray;

public interface NDArray {

    NDArray transpose();

    long[] shape();

    NDArray reshape(char order, int... shape);

    int rank();

    NDArray mmul(NDArray other);

    NDArray mul(NDArray other);

    NDArray div(NDArray other);

    NDArray add(NDArray other);

    NDArray sub(NDArray other);

    NDArray copy(NDArray other);

    NDArray broadcast(long... shape);

}
