package com.hello.tensorflowdemo

import android.graphics.Bitmap
import android.graphics.BitmapFactory
import com.wentaibao.facerecognition.tensorflow.BitmapUtil
import com.wentaibao.facerecognition.tensorflow.FaceTF
import com.wentaibao.facerecognition.tensorflow.MTCNN
import com.wentaibao.facerecognition.tensorflow.MtcnnUtils

class TestThread : Thread() {
    val path = "/sdcard/Pictures/compare/"

    override fun run() {
        super.run()
        val featureMap = mutableMapOf<String, FloatArray>()
        for (i in 1..46) {
            val name = "A$i.jpg"
            val feature = feature(name) ?: continue
            featureMap[name] = feature
        }
        println("featureMap init complete: ${featureMap.size}")

        val featureMap2 = mutableMapOf<String, FloatArray>()
        for (i in 1..46) {
            val name = "B$i.jpg"
            val feature = feature(name) ?: continue
            featureMap2[name] = feature
        }
        println("featureMap2 init complete: ${featureMap2.size}")

        featureMap.forEach { fa1 ->
            featureMap2.forEach { fa2 ->
                val distance = getDistance(fa1.value, fa2.value)
                println("${fa1.key} => ${fa2.key} : $distance")
            }
        }

    }

    fun feature(name: String): FloatArray? {
        println("feature $name start...")
        val bm = BitmapFactory.decodeFile(path + name)
        val face = getFaces(bm)[0] ?: return null
        val feature = getFeature(face)
        println("feature $name success...")
        return feature
    }

    /**
     * MTCNN裁剪出所有人脸并预处理
     */
    fun getFaces(bmp: Bitmap): Array<Bitmap?> {
        var bm = MtcnnUtils.copyBitmap(bmp)
        var faceBitmaps: Array<Bitmap?> = arrayOf(bmp)
        try {
            val boxes = MTCNN.detectFaces(bm, 40)
            faceBitmaps = arrayOfNulls(boxes.size)
            for (i in 0 until boxes.size) {
                // 人脸预处理，设定边距并缩放到160*160
                val bm1 = BitmapUtil.cropBitmap(bm, boxes[i].transform2Rect(), 22)
                faceBitmaps[i] = BitmapUtil.scaleBitmap(bm1, 160, 160)

                // 人脸上描点
                MtcnnUtils.drawRect(bm, boxes[i].transform2Rect())
                MtcnnUtils.drawPoints(bm, boxes[i].landmark)
            }
        } catch (e: Exception) {
            println("mtcnn detect false:$e")
        }
        return faceBitmaps
    }

    /**
     * 提取特征
     */
    fun getFeature(face: Bitmap): FloatArray? {
        return FaceTF.recognizeImage(face)
    }

    /**
     * 计算距离
     */
    fun getDistance(feature1: FloatArray, feature2: FloatArray): Double {
        return FaceTF.sim_distance(feature1, feature2)
    }

}