package com.wentaibao.facerecognition.tensorflow

import android.graphics.*
import android.graphics.drawable.Drawable
import kotlin.math.max
import kotlin.math.min


object BitmapUtil {

    /**
     * 裁剪Bitmap指定矩形区域
     *
     * @param bmp    原图
     * @param face   人脸
     * @param margin 裁剪扩大边距
     * @return 裁剪后的图片
     */
    fun cropBitmap(bmp: Bitmap, face: Rect, margin: Int): Bitmap {
        face.left = max(face.left - margin, 0)
        face.top = max(face.top - margin, 0)
        face.right = min(face.right + margin, bmp.width)
        face.bottom = min(face.bottom + margin, bmp.height)
        val w = face.right - face.left
        val h = face.bottom - face.top
        return Bitmap.createBitmap(bmp, face.left, face.top, w, h)
    }

    /**
     * 缩放Bitmap
     */
    fun scaleBitmap(bm: Bitmap, width: Int, height: Int): Bitmap {
        //        Bitmap bm = MtcnnUtils.copyBitmap(bmp);
        val w = bm.width
        val h = bm.height
        // 计算缩放比例
        val scaleWidth = width.toFloat() / w
        val scaleHeight = height.toFloat() / h
        // 取得想要缩放的matrix参数
        val matrix = Matrix()
        matrix.postScale(scaleWidth, scaleHeight)
        return Bitmap.createBitmap(bm, 0, 0, w, h, matrix, false)
    }

    fun drawableToBitmap(drawable: Drawable): Bitmap {
        // 取 drawable 的长宽
        val w = drawable.intrinsicWidth
        val h = drawable.intrinsicHeight

        // 取 drawable 的颜色格式
        val config = if (drawable.opacity != PixelFormat.OPAQUE)
            Bitmap.Config.ARGB_8888
        else
            Bitmap.Config.RGB_565
        // 建立对应 bitmap
        val bitmap = Bitmap.createBitmap(w, h, config)
        // 建立对应 bitmap 的画布
        val canvas = Canvas(bitmap)
        drawable.setBounds(0, 0, w, h)
        // 把 drawable 内容画到画布中
        drawable.draw(canvas)
        return bitmap
    }
}
