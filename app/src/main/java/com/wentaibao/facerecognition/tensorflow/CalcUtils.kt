package com.wentaibao.facerecognition.tensorflow

import kotlin.math.abs
import kotlin.math.sqrt

object CalcUtils {

    @JvmStatic
    fun main(args: Array<String>) {
        val testData = doubleArrayOf(1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0)
        println("最大值：" + getMax(testData))
        println("最小值：" + getMin(testData))
        println("计数：" + getCount(testData))
        println("求和：" + getSum(testData))
        println("求平均：" + getAverage(testData))
        println("方差：" + getVariance(testData))
        println("标准差：" + getStandardDiviation(testData))

    }

    /**
     * 求给定双精度数组中值的最大值
     *
     * @param inputData 输入数据数组
     * @return 运算结果, 如果输入值不合法，返回为-1
     */
    fun getMax(inputData: DoubleArray?): Double {
        if (inputData == null || inputData.isEmpty())
            return -1.0
        val len = inputData.size
        var max = inputData[0]
        for (i in 0 until len) {
            if (max < inputData[i])
                max = inputData[i]
        }
        return max
    }

    /**
     * 求给定双精度数组中值的最大值
     *
     * @param inputData 输入数据数组
     * @return 运算结果, 如果输入值不合法，返回为-1
     */
    fun getMax(inputData: IntArray?): Int {
        if (inputData == null || inputData.isEmpty())
            return -1
        val len = inputData.size
        var max = inputData[0]
        for (i in 0 until len) {
            if (max < inputData[i])
                max = inputData[i]
        }
        return max
    }

    /**
     * 求求给定双精度数组中值的最小值
     *
     * @param inputData 输入数据数组
     * @return 运算结果, 如果输入值不合法，返回为-1
     */
    fun getMin(inputData: DoubleArray?): Double {
        if (inputData == null || inputData.isEmpty())
            return -1.0
        val len = inputData.size
        var min = inputData[0]
        for (i in 0 until len) {
            if (min > inputData[i])
                min = inputData[i]
        }
        return min
    }

    /**
     * 求求给定双精度数组中值的最小值
     *
     * @param inputData 输入数据数组
     * @return 运算结果, 如果输入值不合法，返回为-1
     */
    fun getMin(inputData: IntArray?): Int {
        if (inputData == null || inputData.isEmpty())
            return -1
        val len = inputData.size
        var min = inputData[0]
        for (i in 0 until len) {
            if (min > inputData[i])
                min = inputData[i]
        }
        return min
    }

    /**
     * 求给定双精度数组中值的和
     *
     * @param inputData 输入数据数组
     * @return 运算结果
     */
    fun getSum(inputData: DoubleArray?): Double {
        if (inputData == null || inputData.isEmpty())
            return -1.0
        val len = inputData.size
        var sum = 0.0
        for (i in 0 until len) {
            sum += inputData[i]
        }
        return sum
    }

    /**
     * 求给定双精度数组中值的和
     *
     * @param inputData 输入数据数组
     * @return 运算结果
     */
    fun getSum(inputData: IntArray?): Long {
        if (inputData == null || inputData.isEmpty())
            return -1
        val len = inputData.size
        var sum: Long = 0
        for (i in 0 until len) {
            sum += inputData[i]
        }
        return sum

    }

    /**
     * 求给定双精度数组中值的数目
     *
     * @param inputData 输入数据数组
     * @return 运算结果
     */
    fun getCount(inputData: DoubleArray?): Int {
        return inputData?.size ?: -1
    }

    /**
     * 求给定双精度数组中值的数目
     *
     * @param inputData 输入数据数组
     * @return 运算结果
     */
    fun getCount(inputData: IntArray?): Int {
        return inputData?.size ?: -1
    }

    /**
     * 求给定双精度数组中值的平均值
     *
     * @param inputData 输入数据数组
     * @return 运算结果
     */
    fun getAverage(inputData: DoubleArray?): Double {
        if (inputData == null || inputData.isEmpty()) return -1.0
        val len = inputData.size
        val result: Double
        result = getSum(inputData) / len
        return result
    }

    /**
     * 求给定双精度数组中值的平均值
     *
     * @param inputData 输入数据数组
     * @return 运算结果
     */
    fun getAverage(inputData: IntArray?): Double {
        if (inputData == null || inputData.isEmpty()) return -1.0
        val len = inputData.size
        val result: Double
        result = getSum(inputData) * 1.0 / len
        return result
    }

    /**
     * 求给定双精度数组中值的平方和
     *
     * @param inputData 输入数据数组
     * @return 运算结果
     */
    fun getSquareSum(inputData: DoubleArray?): Double {
        if (inputData == null || inputData.isEmpty()) return -1.0
        val len = inputData.size
        var sqrsum = 0.0
        for (i in 0 until len) {
            sqrsum += inputData[i] * inputData[i]
        }
        return sqrsum
    }

    /**
     * 求给定双精度数组中值的平方和
     *
     * @param inputData 输入数据数组
     * @return 运算结果
     */
    fun getSquareSum(inputData: IntArray?): Long {
        if (inputData == null || inputData.isEmpty()) return -1
        val len = inputData.size
        var sqrsum: Long = 0
        for (i in 0 until len) {
            sqrsum += inputData[i] * inputData[i]
        }
        return sqrsum
    }

    /**
     * 求给定双精度数组中值的方差
     *
     * @param inputData 输入数据数组
     * @return 运算结果
     */
    fun getVariance(inputData: DoubleArray): Double {
        val count = getCount(inputData)
        val sqrsum = getSquareSum(inputData)
        val average = getAverage(inputData)
        val result: Double
        result = (sqrsum - count.toDouble() * average * average) / count
        return result
    }

    /**
     * 求给定双精度数组中值的方差
     *
     * @param inputData 输入数据数组
     * @return 运算结果
     */
    fun getVariance(inputData: IntArray): Double {
        val count = getCount(inputData)
        val sqrsum = getSquareSum(inputData)
        val average = getAverage(inputData)
        val result: Double
        result = (sqrsum - count.toDouble() * average * average) / count
        return result
    }

    /**
     * 求给定双精度数组中值的标准差
     *
     * @param inputData 输入数据数组
     * @return 运算结果
     */
    fun getStandardDiviation(inputData: DoubleArray): Double {
        //绝对值化很重要
        return sqrt(abs(getVariance(inputData)))
    }

    /**
     * 求给定双精度数组中值的标准差
     *
     * @param inputData 输入数据数组
     * @return 运算结果
     */
    fun getStandardDiviation(inputData: IntArray): Double {
        //绝对值化很重要
        return sqrt(abs(getVariance(inputData)))
    }
}
