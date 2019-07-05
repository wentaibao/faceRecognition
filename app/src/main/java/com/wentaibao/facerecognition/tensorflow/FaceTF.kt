package com.wentaibao.facerecognition.tensorflow

import android.content.res.AssetManager
import android.graphics.Bitmap
import android.graphics.Color
import android.util.Log
import org.tensorflow.contrib.android.TensorFlowInferenceInterface

import java.io.FileInputStream
import java.io.IOException
import java.nio.ByteBuffer
import kotlin.math.max
import kotlin.math.pow
import kotlin.math.sqrt

object FaceTF {
    private val TAG = FaceTF::class.java.simpleName

    // tensor name
    private val INPUT_NAME = "input:0"
    private val OUTPUT_NAME = "embeddings:0"
    private val PHASE_NAME = "phase_train:0"
    //神经网络输入大小
    private val INPUT_SIZE = 160L

    private lateinit var tensorFlowInference: TensorFlowInferenceInterface

    /**
     * @param assetManager
     * @param modelName = "file:///android_asset/face_model.pb"
     */
    fun loadModel(assetManager: AssetManager, modelName: String) {
        tensorFlowInference = TensorFlowInferenceInterface(assetManager, modelName)
    }

    /**
     * @param modelPath 外部存储器上model文件路径
     */
    fun loadModel(modelPath: String) {
        try {
            val t = System.currentTimeMillis()
            val fis = FileInputStream(modelPath)
            tensorFlowInference = TensorFlowInferenceInterface(fis)
            Log.d(TAG, "loadModel: Successful $modelPath")
            Log.d(TAG, "loadModel time: " + (System.currentTimeMillis() - t))
            //            Iterator<Operation> iterator = tensorFlowInference.graph().operations();
            //            while (iterator.hasNext()) {
            //                Log.d(TAG, "Operation name: " + iterator.next().name());
            //            }
            fis.close()
        } catch (e: IOException) {
            e.printStackTrace()
        }

    }

    /**
     * 提取图片特征
     * @param bitmap 处理好的160*160人脸图片
     * @return float[512]
     */
    fun recognizeImage(bitmap: Bitmap): FloatArray? {
        if (tensorFlowInference == null) return null

        val t = System.currentTimeMillis()
        val w = bitmap.width
        val h = bitmap.height
        Log.d(TAG, "recognizeImage: $w * $h")

        //(0)图片预处理，normalizeImage
        //        float[] values = normalizeImage(bitmap);
        val values = normalizeImage3(bitmap)
        //        byte[] bts = bitmap2RGB(bitmap);
        //        float[] values = new float[bts.length];
        //        for (int i = 0; i < bts.length; i++) {
        //            values[i] = bts[i];
        //        }
        //(1)Feed
        try {
            // Expects arg[0] to be float 输入输出都为 float
            tensorFlowInference.feed(INPUT_NAME, values, 1L, INPUT_SIZE, INPUT_SIZE, 3L)
            val phase = BooleanArray(1)
            phase[0] = false
            tensorFlowInference.feed(PHASE_NAME, phase)
        } catch (e: Exception) {
            Log.e(TAG, "feed Error: ", e)
            return null
        }

        //(2)run
        try {
            tensorFlowInference.run(arrayOf(OUTPUT_NAME), false)
        } catch (e: Exception) {
            Log.e(TAG, "run error: ", e)
            return null
        }

        //(3)fetch
        val outputs = FloatArray(512)
        try {
            tensorFlowInference.fetch(OUTPUT_NAME, outputs)
        } catch (e: Exception) {
            Log.e(TAG, "fetch error: ", e)
            return null
        }

        Log.i(TAG, "recognizeImage time: " + (System.currentTimeMillis() - t))
        return outputs
    }

    //读取Bitmap像素值，预处理(-127.5 /128)，转化为一维数组返回
    fun normalizeImage(bitmap: Bitmap): FloatArray {
        val w = bitmap.width
        val h = bitmap.height
        val floatValues = FloatArray(w * h * 3)
        val intValues = IntArray(w * h)
        bitmap.getPixels(intValues, 0, bitmap.width, 0, 0, bitmap.width, bitmap.height)
        // 将像素映射到[-1,1]区间内
        val imageMean = 127.5f
        val imageStd = 128f
        for (i in intValues.indices) {
            val `val` = intValues[i]
            floatValues[i * 3 + 0] = ((`val` shr 16 and 0xFF) - imageMean) / imageStd
            floatValues[i * 3 + 1] = ((`val` shr 8 and 0xFF) - imageMean) / imageStd
            floatValues[i * 3 + 2] = ((`val` and 0xFF) - imageMean) / imageStd
        }
        return floatValues
    }

    fun normalizeImage2(bitmap: Bitmap): FloatArray {
        val w = bitmap.width
        val h = bitmap.height
        val intValues = IntArray(w * h)
        bitmap.getPixels(intValues, 0, bitmap.width, 0, 0, w, h)
        val floatValues = FloatArray(w * h * 3)
        for (i in intValues.indices) { // 分别取GRB
            val `val` = intValues[i]
            floatValues[i * 3 + 0] = (`val` shr 16 and 0xFF).toFloat()
            floatValues[i * 3 + 1] = (`val` shr 8 and 0xFF).toFloat()
            floatValues[i * 3 + 2] = (`val` and 0xFF).toFloat()
        }
        return floatValues
    }

    /**
     * 取图片RGB数组并处理成float数组
     *
     *
     * python 代码
     * def prewhiten(x):{
     * mean = np.mean(x)
     * * std = np.std(x)
     * * std_adj = np.maximum(std, 1.0/np.sqrt(x.size))
     * * y = np.multiply(np.subtract(x, mean), 1/std_adj)
     * * return y
     * }
     * x:RGB 整型
     * mean: 求平均值
     * std: 求标准差
     * std_adj: 取较大值 sqrt:开方
     * multiply:乘 subtract:减
     */
    fun normalizeImage3(bitmap: Bitmap): FloatArray {
        val w = bitmap.width
        val h = bitmap.height
        val pixels = IntArray(w * h)
        bitmap.getPixels(pixels, 0, bitmap.width, 0, 0, w, h)
        val intValues = IntArray(w * h * 3)
        for (i in pixels.indices) { // 分别取GRB
            val `val` = pixels[i]
            intValues[i * 3 + 0] = `val` shr 16 and 0xFF
            intValues[i * 3 + 1] = `val` shr 8 and 0xFF
            intValues[i * 3 + 2] = `val` and 0xFF
        }

        val avg = CalcUtils.getAverage(intValues) // 平均值
        val std = CalcUtils.getStandardDiviation(intValues) // 标准差
        val std_max = max(std, 1.0 / sqrt(intValues.size.toDouble()))

        val subtractValues = DoubleArray(intValues.size) // 各元素之差
        for (i in subtractValues.indices) {
            subtractValues[i] = intValues[i] - avg
        }

        val reciprocal = 1.0 / std_max
        val multiplyValues = DoubleArray(intValues.size) // 各元素乘积
        for (i in multiplyValues.indices) {
            multiplyValues[i] = subtractValues[i] * reciprocal
        }

        val floatValues = FloatArray(intValues.size)
        for (i in floatValues.indices) {
            floatValues[i] = multiplyValues[i].toFloat()
        }
        return floatValues
    }

    fun bitmap2RGB(bitmap: Bitmap): ByteArray {
        val bytes = bitmap.byteCount  //返回可用于储存此位图像素的最小字节数
        val buffer = ByteBuffer.allocate(bytes) //  使用allocate()静态方法创建字节缓冲区
        bitmap.copyPixelsToBuffer(buffer) // 将位图的像素复制到指定的缓冲区
        val rgba = buffer.array()
        val pixels = ByteArray(rgba.size / 4 * 3)
        val count = rgba.size / 4
        //Bitmap像素点的色彩通道排列顺序是RGBA
        for (i in 0 until count) {
            pixels[i * 3] = rgba[i * 4]        //R
            pixels[i * 3 + 1] = rgba[i * 4 + 1]    //G
            pixels[i * 3 + 2] = rgba[i * 4 + 2]       //B

        }
        return pixels
    }

    /**
     * RGB 值正常 但是负值
     */
    fun getRGBFromBitmap(bmp: Bitmap): ByteArray {
        val w = bmp.width
        val h = bmp.height
        val pixels = ByteArray(w * h * 3) // Allocate for RGB
        var k = 0
        for (x in 0 until h) {
            for (y in 0 until w) {
                val color = bmp.getPixel(y, x)
                pixels[k * 3] = Color.red(color).toByte()
                pixels[k * 3 + 1] = Color.green(color).toByte()
                pixels[k * 3 + 2] = Color.blue(color).toByte()
                k++
            }
        }
        return pixels
    }


    /**
     * 两个向量可以为任意维度，但必须保持维度相同，表示n维度中的两点
     * https://blog.csdn.net/C_son/article/details/43889195
     *
     * @return 两点间距离
     */
    fun sim_distance(vector1: FloatArray, vector2: FloatArray): Double {
        var distance = 0.0
        if (vector1.size == vector2.size) {
            for (i in vector1.indices) {
                val temp = (vector1[i] - vector2[i]).toDouble().pow(2.0)
                distance += temp
            }
            distance = sqrt(distance)
        }
        return distance
    }

}
