package com.wentaibao.facerecognition.tensorflow

/**
 * MTCNN For Android
 * by cjf@xmu 20180625
 * https://github.com/vcvycy/MTCNN4Android
 */

import android.graphics.*
import android.util.Log
import java.util.*

object MtcnnUtils {
    //复制图片，并设置isMutable=true
    fun copyBitmap(bitmap: Bitmap): Bitmap {
        return bitmap.copy(bitmap.config, true)
    }

    //在bitmap中画矩形
    fun drawRect(bitmap: Bitmap, rect: Rect) {
        try {
            val canvas = Canvas(bitmap)
            val paint = Paint()
            val r = 255//(int)(Math.random()*255);
            val g = 0//(int)(Math.random()*255);
            val b = 0//(int)(Math.random()*255);
            paint.color = Color.rgb(r, g, b)
            paint.strokeWidth = (1 + bitmap.width / 500).toFloat()
            paint.style = Paint.Style.STROKE
            canvas.drawRect(rect, paint)
        } catch (e: Exception) {
            Log.i("MtcnnUtils", "[*] error$e")
        }

    }

    //在图中画点
    fun drawPoints(bitmap: Bitmap, landmark: Array<Point>) {
        for (i in landmark.indices) {
            val x = landmark[i].x
            val y = landmark[i].y
            //Log.i("MtcnnUtils","[*] landmarkd "+x+ "  "+y);
            drawRect(bitmap, Rect(x - 1, y - 1, x + 1, y + 1))
        }
    }

    //Flip alone diagonal
    //对角线翻转。data大小原先为h*w*stride，翻转后变成w*h*stride
    fun flip_diag(data: FloatArray, h: Int, w: Int, stride: Int) {
        val tmp = FloatArray(w * h * stride)
        for (i in 0 until w * h * stride) tmp[i] = data[i]
        for (y in 0 until h)
            for (x in 0 until w) {
                for (z in 0 until stride)
                    data[(x * h + y) * stride + z] = tmp[(y * w + x) * stride + z]
            }
    }

    //src转为二维存放到dst中
    fun expand(src: FloatArray, dst: Array<FloatArray>) {
        var idx = 0
        for (y in dst.indices)
            for (x in 0 until dst[0].size)
                dst[y][x] = src[idx++]
    }

    //src转为三维存放到dst中
    fun expand(src: FloatArray, dst: Array<Array<FloatArray>>) {
        var idx = 0
        for (y in dst.indices)
            for (x in 0 until dst[0].size)
                for (c in 0 until dst[0][0].size)
                    dst[y][x][c] = src[idx++]

    }

    //dst=src[:,:,1]
    fun expandProb(src: FloatArray, dst: Array<FloatArray>) {
        var idx = 0
        for (y in dst.indices)
            for (x in 0 until dst[0].size)
                dst[y][x] = src[idx++ * 2 + 1]
    }

    //box转化为rect
    fun boxes2rects(boxes: Vector<Box>): Array<Rect?> {
        var cnt = 0
        for (i in boxes.indices) if (!boxes[i].deleted) cnt++
        val r = arrayOfNulls<Rect>(cnt)
        var idx = 0
        for (i in boxes.indices)
            if (!boxes[i].deleted)
                r[idx++] = boxes[i].transform2Rect()
        return r
    }

    //删除做了delete标记的box
    fun updateBoxes(boxes: Vector<Box>): Vector<Box> {
        val b = Vector<Box>()
        for (i in boxes.indices)
            if (!boxes[i].deleted)
                b.addElement(boxes[i])
        return b
    }

    //
    fun showPixel(v: Int) {
        Log.i("MainActivity", "[*]Pixel:R" + (v shr 16 and 0xff) + "G:" + (v shr 8 and 0xff) + " B:" + (v and 0xff))
    }
}