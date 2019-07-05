package com.wentaibao.facerecognition.tensorflow

import android.content.Context
import android.graphics.Bitmap
import android.graphics.PointF
import android.graphics.Rect
import android.media.FaceDetector

object FaceManger {

    val faceModel = "file:///android_asset/face_model.pb" // 加载assets目录下model
    val mtcnnModel = "file:///android_asset/mtcnn_freezed_model.pb"

    /**
     * 加载模型
     */
    fun initModel(context: Context) {
        MTCNN.loadModel(context.assets, mtcnnModel)
        FaceTF.loadModel(context.assets, faceModel)
    }

    /**
     * 检测人脸，返回裁剪出的人脸
     */
    fun detectFaceMTCNN(bmp: Bitmap): DetectFaceResult {
        val result = DetectFaceResult(bmp, 0, mutableListOf())
        var bm = MtcnnUtils.copyBitmap(bmp)
        try {
            val minFaceSize = 40
            val boxes = MTCNN.detectFaces(bm, minFaceSize)
            result.faceCount = boxes.size
            for (i in 0 until boxes.size) {
                // 裁剪人脸
                val face = BitmapUtil.cropBitmap(bm, boxes[i].transform2Rect(), 22)
                val squareFace = BitmapUtil.scaleBitmap(face, 160, 160)
                result.faces.add(squareFace)

                // 画人脸框
                MtcnnUtils.drawRect(bm, boxes[i].transform2Rect())
                MtcnnUtils.drawPoints(bm, boxes[i].landmark)
                result.faceBm = bm
            }
        } catch (e: Exception) {
            println("mtcnn detect false:$e")
        }
        return result
    }

    /**
     * Android自带人脸检测，速度更快
     */
    fun detectFaceAndroid(bmp: Bitmap): DetectFaceResult {
        val result = DetectFaceResult(bmp, 0, mutableListOf())
//        var bm = MtcnnUtils.copyBitmap(bmp)
        var bm565 = bmp.copy(Bitmap.Config.RGB_565, true)
        val maxFaceCount = 12
        var faces = arrayOfNulls<FaceDetector.Face>(maxFaceCount)
        val faceDetector = FaceDetector(bm565.width, bm565.height, maxFaceCount)
        val faceCount = faceDetector.findFaces(bm565, faces)
        println("检测到${faceCount}张人脸")
        if (faceCount < 1) return result

        result.faceCount = faceCount
        for (i in 0 until faceCount) {
            val myMidPoint = PointF()
            faces[i]!!.getMidPoint(myMidPoint)
            val myEyesDistance = faces[i]!!.eyesDistance()
            val rect = Rect(
                (myMidPoint.x - myEyesDistance * 1.5).toInt(),
                (myMidPoint.y - myEyesDistance * 1.5).toInt(),
                (myMidPoint.x + myEyesDistance * 1.5).toInt(),
                (myMidPoint.y + myEyesDistance * 1.8).toInt()
            )

            // 裁剪人脸
            val face = BitmapUtil.cropBitmap(bm565, rect, 0)
            val squareFace = BitmapUtil.scaleBitmap(face, 160, 160)
            result.faces.add(squareFace)

            // 画人脸框
            MtcnnUtils.drawRect(bm565, rect)
            result.faceBm = bm565
        }

        return result
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

    /**
     * 欧式距离转相似度
     * Excel 计算出的趋势函数
     * y = -6.6291x3 - 16.753x2 - 5.9076x + 100.14
     * R2 = 0.9983
     */
    fun distanceToSimilarity(distance: Double): Double {
        return -6.6291 * distance * distance * distance - 16.753 * distance * distance - 5.9076 * distance + 100
    }

    data class DetectFaceResult(var faceBm: Bitmap, var faceCount: Int, var faces: MutableList<Bitmap>)
}