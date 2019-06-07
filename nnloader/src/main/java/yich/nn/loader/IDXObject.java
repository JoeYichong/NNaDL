package yich.nn.loader;

import yich.base.dbc.Require;
import yich.base.util.BitUtil;

import java.util.Arrays;

public class IDXObject {
    byte data_type;
    int[] dim_sizes;

    // Pixels are organized row-wise.
    // Pixel values are 0 to 255. 0 means background (white), 255 means foreground (black).
    byte[] data;

    public IDXObject(byte data_type, int[] dim_sizes, byte[] data) {
        Require.argumentNotNull(dim_sizes, "dim_sizes");
        Require.argumentWCM(dim_sizes.length > 0,"int[] 'dim_sizes' is empty.");
        Require.argumentNotNull(data, "data");
        Require.argumentWCM(data.length > 0,"byte[] 'data' is empty.");
        Require.argumentWCM(dataLen0(data_type) > 0,
                "Unknown Data Type: " + BitUtil.getBitString(data_type));
        Require.argumentWCM(check(dim_sizes, data),
                "Data doesn't match the dimension size: " + Arrays.toString(dim_sizes));

        this.data_type = data_type;
        this.dim_sizes = dim_sizes;
        this.data = data;
    }

    private boolean check(int[] d_sizes, byte[] data) {
        int total = d_sizes[0];
        for (int i = 1; i < d_sizes.length; i++) {
            total *= d_sizes[i];
        }
        return total == data.length;
    }

    private int dataLen0(byte type) {
        switch (type) {
            case 0x08:           // unsigned byte
            case 0x09: return 1; // signed byte
            case 0x0B: return 2; // short (2 bytes)
            case 0x0C:           // int (4 bytes)
            case 0x0D: return 4; // float (4 bytes)
            case 0x0E: return 8; // double (8 bytes)
            default: return 0;
        }
    }

    public int dataLen() {
        return dataLen0(this.data_type);
    }

    public int blockSize() {
        int total = dataLen();
        for (int i = 1; i < dim_sizes.length; i++) {
            total *= dim_sizes[i];
        }
        return total;
    }

    public byte getDataType() {
        return data_type;
    }

    public int[] getDimSizes() {
        return dim_sizes;
    }

    public byte[] getData() {
        return data;
    }

    public double[] getAsDouble(int index) {
        Require.argument(index >= 0, index,"index >= 0");
        Require.argument(index < dim_sizes[0], index,"index < dim_sizes[0]");

        int blockSize = blockSize();
        double[] d_data = new double[blockSize];
        int offset = blockSize * index;
        for (int i = 0; i < blockSize; i++) {
            d_data[i] = (double) (data[offset + i]);
        }
        return d_data;
    }

    /**
     * Get an element of the first dimension using index.
     * */
    public byte[] get(int index) {
        int len = 1;
        for (int i = 1; i < dim_sizes.length; i++) {
            len *= dim_sizes[i];
        }
        return Arrays.copyOfRange(data, len * index, len * (index + 1));
    }

    public void printInfo() {
        System.out.println("dataLen: " + this.dataLen());
        System.out.println("blockSize: " + this.blockSize());
        System.out.println("dataType: " + BitUtil.getBitString(this.getDataType()));
        System.out.println("dimSizes: " + Arrays.toString(this.getDimSizes()));

    }

}
