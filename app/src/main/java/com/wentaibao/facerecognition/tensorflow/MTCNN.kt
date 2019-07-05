package com.wentaibao.facerecognition.tensorflow

/**
 * MTCNN For Android
 * by cjf@xmu 20180625
 * https://github.com/vcvycy/MTCNN4Android
 */

import android.content.res.AssetManager
import android.graphics.Bitmap
import android.graphics.Matrix
import android.graphics.Point
import android.util.Log
import org.tensorflow.contrib.android.TensorFlowInferenceInterface
import java.io.FileInputStream
import java.io.IOException
import java.lang.Math.max
import java.lang.Math.min
import java.util.*
import kotlin.math.ceil
import kotlin.math.roundToInt

object MTCNN {

    private val TAG = MTCNN::class.java.simpleName

    // tensor name
    private val PNetInName = "pnet/input:0"
    private val PNetOutName = arrayOf("pnet/prob1:0", "pnet/conv4-2/BiasAdd:0")
    private val RNetInName = "rnet/input:0"
    private val RNetOutName = arrayOf("rnet/prob1:0", "rnet/conv5-2/conv5-2:0")
    private val ONetInName = "onet/input:0"
    private val ONetOutName = arrayOf("onet/prob1:0", "onet/conv6-2/conv6-2:0", "onet/conv6-3/conv6-3:0")

    // 参数
    private val factor = 0.709f
    private val PNetThreshold = 0.6f
    private val RNetThreshold = 0.7f
    private val ONetThreshold = 0.7f

    private var lastProcessTime: Long = 0   //最后一张图片处理的时间ms
    private lateinit var inferenceInterface: TensorFlowInferenceInterface

    fun loadModel(modelPath: String) {
        try {
            val t = System.currentTimeMillis()
            val fis = FileInputStream(modelPath)
            inferenceInterface = TensorFlowInferenceInterface(fis)
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

    fun loadModel(assetManager: AssetManager, modelPath: String) {
        inferenceInterface = TensorFlowInferenceInterface(assetManager, modelPath)
    }

    //读取Bitmap像素值，预处理(-127.5 /128)，转化为一维数组返回
    private fun normalizeImage(bitmap: Bitmap): FloatArray {
        val w = bitmap.width
        val h = bitmap.height
        val floatValues = FloatArray(w * h * 3)
        val intValues = IntArray(w * h)
        bitmap.getPixels(intValues, 0, bitmap.width, 0, 0, bitmap.width, bitmap.height)
        val imageMean = 127.5f
        val imageStd = 128f

        for (i in intValues.indices) {
            val rgb = intValues[i]
            floatValues[i * 3 + 0] = ((rgb shr 16 and 0xFF) - imageMean) / imageStd
            floatValues[i * 3 + 1] = ((rgb shr 8 and 0xFF) - imageMean) / imageStd
            floatValues[i * 3 + 2] = ((rgb and 0xFF) - imageMean) / imageStd
        }
        return floatValues
    }

    /*
       检测人脸,minSize是最小的人脸像素值
     */
    private fun bitmapResize(bm: Bitmap, scale: Float): Bitmap {
        val width = bm.width
        val height = bm.height
        // CREATE A MATRIX FOR THE MANIPULATION。matrix指定图片仿射变换参数
        val matrix = Matrix()
        // RESIZE THE BIT MAP
        matrix.postScale(scale, scale)
        return Bitmap.createBitmap(bm, 0, 0, width, height, matrix, true)
    }

    //输入前要翻转，输出也要翻转
    private fun PNetForward(
        bitmap: Bitmap,
        PNetOutProb: Array<FloatArray>,
        PNetOutBias: Array<Array<FloatArray>>
    ) {
        inferenceInterface?.let {
            val w = bitmap.width
            val h = bitmap.height

            val PNetIn = normalizeImage(bitmap)
            MtcnnUtils.flip_diag(PNetIn, h, w, 3) //沿着对角线翻转
            it.feed(PNetInName, PNetIn, 1L, w.toLong(), h.toLong(), 3L)
            it.run(PNetOutName, false)
            val PNetOutSizeW = ceil(w * 0.5 - 5).toInt()
            val PNetOutSizeH = ceil(h * 0.5 - 5).toInt()
            val PNetOutP = FloatArray(PNetOutSizeW * PNetOutSizeH * 2)
            val PNetOutB = FloatArray(PNetOutSizeW * PNetOutSizeH * 4)
            it.fetch(PNetOutName[0], PNetOutP)
            it.fetch(PNetOutName[1], PNetOutB)
            //【写法一】先翻转，后转为2/3维数组
            MtcnnUtils.flip_diag(PNetOutP, PNetOutSizeW, PNetOutSizeH, 2)
            MtcnnUtils.flip_diag(PNetOutB, PNetOutSizeW, PNetOutSizeH, 4)
            MtcnnUtils.expand(PNetOutB, PNetOutBias)
            MtcnnUtils.expandProb(PNetOutP, PNetOutProb)
            /*
            *【写法二】这个比较快，快了3ms。意义不大，用上面的方法比较直观
            for (int y=0;y<PNetOutSizeH;y++)
                for (int x=0;x<PNetOutSizeW;x++){
                   int idx=PNetOutSizeH*x+y;
                   PNetOutProb[y][x]=PNetOutP[idx*2+1];
                   for(int i=0;i<4;i++)
                       PNetOutBias[y][x][i]=PNetOutB[idx*4+i];
                }
            */
        }
    }

    //Non-Maximum Suppression
    //nms，不符合条件的deleted设置为true
    private fun nms(boxes: Vector<Box>, threshold: Float, method: String) {
        //NMS.两两比对
        //int delete_cnt=0;
        val cnt = 0
        for (i in boxes.indices) {
            val box = boxes[i]
            if (!box.deleted) {
                //score<0表示当前矩形框被删除
                for (j in i + 1 until boxes.size) {
                    val box2 = boxes[j]
                    if (!box2.deleted) {
                        val x1 = max(box.box[0], box2.box[0])
                        val y1 = max(box.box[1], box2.box[1])
                        val x2 = min(box.box[2], box2.box[2])
                        val y2 = min(box.box[3], box2.box[3])
                        if (x2 < x1 || y2 < y1) continue
                        val areaIoU = (x2 - x1 + 1) * (y2 - y1 + 1)
                        var iou = 0f
                        if (method == "Union")
                            iou = 1.0f * areaIoU / (box.area() + box2.area() - areaIoU)
                        else if (method == "Min") {
                            iou = 1.0f * areaIoU / min(box.area(), box2.area())
                            Log.i(TAG, "[*]iou=$iou")
                        }
                        if (iou >= threshold) { //删除prob小的那个框
                            if (box.score > box2.score)
                                box2.deleted = true
                            else
                                box.deleted = true
                            //delete_cnt++;
                        }
                    }
                }
            }
        }
        //Log.i(TAG,"[*]sum:"+boxes.size()+" delete:"+delete_cnt);
    }

    private fun generateBoxes(
        prob: Array<FloatArray>,
        bias: Array<Array<FloatArray>>,
        scale: Float,
        threshold: Float,
        boxes: Vector<Box>
    ): Int {
        val h = prob.size
        val w = prob[0].size
        //Log.i(TAG,"[*]height:"+prob.length+" width:"+prob[0].length);
        for (y in 0 until h)
            for (x in 0 until w) {
                val score = prob[y][x]
                //only accept prob >threadshold(0.6 here)
                if (score > threshold) {
                    val box = Box()
                    //score
                    box.score = score
                    //box
                    box.box[0] = (x * 2 / scale).roundToInt()
                    box.box[1] = (y * 2 / scale).roundToInt()
                    box.box[2] = ((x * 2 + 11) / scale).roundToInt()
                    box.box[3] = ((y * 2 + 11) / scale).roundToInt()
                    //bbr
                    for (i in 0..3)
                        box.bbr[i] = bias[y][x][i]
                    //add
                    boxes.addElement(box)
                }
            }
        return 0
    }

    private fun BoundingBoxReggression(boxes: Vector<Box>) {
        for (i in boxes.indices)
            boxes[i].calibrate()
    }

    //Pnet + Bounding Box Regression + Non-Maximum Regression

    /**
     * NMS执行完后，才执行Regression
     * (1) For each scale , use NMS with threshold=0.5
     * (2) For all candidates , use NMS with threshold=0.7
     * (3) Calibrate Bounding Box
     * 注意：CNN输入图片最上面一行，坐标为[0..width,0]。所以Bitmap需要对折后再跑网络;网络输出同理.
     */
    private fun PNet(bitmap: Bitmap, minSize: Int): Vector<Box> {
        val whMin = min(bitmap.width, bitmap.height)
        var currentFaceSize = minSize.toFloat()  //currentFaceSize=minSize/(factor^k) k=0,1,2... until excced whMin
        val totalBoxes = Vector<Box>()
        //【1】Image Paramid and Feed to Pnet
        while (currentFaceSize <= whMin) {
            val scale = 12.0f / currentFaceSize
            //(1)Image Resize
            val bm = bitmapResize(bitmap, scale)
            val w = bm.width
            val h = bm.height
            //(2)RUN CNN
            val PNetOutSizeW = (ceil(w * 0.5 - 5) + 0.5).toInt()
            val PNetOutSizeH = (ceil(h * 0.5 - 5) + 0.5).toInt()
            val PNetOutProb = Array(PNetOutSizeH) { FloatArray(PNetOutSizeW) }
            val PNetOutBias = Array(PNetOutSizeH) { Array(PNetOutSizeW) { FloatArray(4) } }
            PNetForward(bm, PNetOutProb, PNetOutBias)
            //(3)数据解析
            val curBoxes = Vector<Box>()
            generateBoxes(PNetOutProb, PNetOutBias, scale, PNetThreshold, curBoxes)
            //Log.i(TAG,"[*]CNN Output Box number:"+curBoxes.size()+" Scale:"+scale);
            //(4)nms 0.5
            nms(curBoxes, 0.5f, "Union")
            //(5)add to totalBoxes
            for (i in curBoxes.indices)
                if (!curBoxes[i].deleted)
                    totalBoxes.addElement(curBoxes[i])
            //Face Size等比递增
            currentFaceSize /= factor
        }
        //NMS 0.7
        nms(totalBoxes, 0.7f, "Union")
        //BBR
        BoundingBoxReggression(totalBoxes)
        return MtcnnUtils.updateBoxes(totalBoxes)
    }

    private fun crop_and_resize(bitmap: Bitmap, box: Box, size: Int, data: FloatArray) {
        //(2)crop and resize
        val matrix = Matrix()
        val scale = 1.0f * size / box.width()
        matrix.postScale(scale, scale)
        val croped = Bitmap.createBitmap(bitmap, box.left(), box.top(), box.width(), box.height(), matrix, true)
        //(3)save
        val pixels_buf = IntArray(size * size)
        croped.getPixels(pixels_buf, 0, croped.width, 0, 0, croped.width, croped.height)
        val imageMean = 127.5f
        val imageStd = 128f
        for (i in pixels_buf.indices) {
            val rgb = pixels_buf[i]
            data[i * 3 + 0] = ((rgb shr 16 and 0xFF) - imageMean) / imageStd
            data[i * 3 + 1] = ((rgb shr 8 and 0xFF) - imageMean) / imageStd
            data[i * 3 + 2] = ((rgb and 0xFF) - imageMean) / imageStd
        }
    }

    /*
     * RNET跑神经网络，将score和bias写入boxes
     */
    private fun RNetForward(RNetIn: FloatArray, boxes: Vector<Box>) {
        inferenceInterface?.let {
            val num = RNetIn.size / 24 / 24 / 3
            //feed & run
            it.feed(RNetInName, RNetIn, num.toLong(), 24L, 24L, 3L)
            it.run(RNetOutName, false)
            //fetch
            val RNetP = FloatArray(num * 2)
            val RNetB = FloatArray(num * 4)
            it.fetch(RNetOutName[0], RNetP)
            it.fetch(RNetOutName[1], RNetB)
            //转换
            for (i in 0 until num) {
                boxes[i].score = RNetP[i * 2 + 1]
                for (j in 0..3)
                    boxes[i].bbr[j] = RNetB[i * 4 + j]
            }
        }
    }

    //Refine Net
    private fun RNet(bitmap: Bitmap, boxes: Vector<Box>): Vector<Box> {
        //RNet Input Init
        val num = boxes.size
        val RNetIn = FloatArray(num * 24 * 24 * 3)
        val curCrop = FloatArray(24 * 24 * 3)
        var RNetInIdx = 0
        for (i in 0 until num) {
            crop_and_resize(bitmap, boxes[i], 24, curCrop)
            MtcnnUtils.flip_diag(curCrop, 24, 24, 3)
            //Log.i(TAG,"[*]Pixels values:"+curCrop[0]+" "+curCrop[1]);
            for (j in curCrop.indices) RNetIn[RNetInIdx++] = curCrop[j]
        }
        //Run RNet
        RNetForward(RNetIn, boxes)
        //RNetThreshold
        for (i in 0 until num)
            if (boxes[i].score < RNetThreshold)
                boxes[i].deleted = true
        //Nms
        nms(boxes, 0.7f, "Union")
        BoundingBoxReggression(boxes)
        return MtcnnUtils.updateBoxes(boxes)
    }

    /*
     * ONet跑神经网络，将score和bias写入boxes
     */
    private fun ONetForward(ONetIn: FloatArray, boxes: Vector<Box>) {
        inferenceInterface?.let {
            val num = ONetIn.size / 48 / 48 / 3
            //feed & run
            it.feed(ONetInName, ONetIn, num.toLong(), 48L, 48L, 3L)
            it.run(ONetOutName, false)
            //fetch
            val ONetP = FloatArray(num * 2) //prob
            val ONetB = FloatArray(num * 4) //bias
            val ONetL = FloatArray(num * 10) //landmark
            it.fetch(ONetOutName[0], ONetP)
            it.fetch(ONetOutName[1], ONetB)
            it.fetch(ONetOutName[2], ONetL)
            //转换
            for (i in 0 until num) {
                //prob
                boxes[i].score = ONetP[i * 2 + 1]
                //bias
                for (j in 0..3)
                    boxes[i].bbr[j] = ONetB[i * 4 + j]

                //landmark
                for (j in 0..4) {
                    val x = boxes[i].left() + (ONetL[i * 10 + j] * boxes[i].width()).toInt()
                    val y = boxes[i].top() + (ONetL[i * 10 + j + 5] * boxes[i].height()).toInt()
                    boxes[i].landmark[j] = Point(x, y)
                    //Log.i(TAG,"[*] landmarkd "+x+ "  "+y);
                }
            }
        }
    }

    //ONet
    private fun ONet(bitmap: Bitmap, boxes: Vector<Box>): Vector<Box> {
        //ONet Input Init
        val num = boxes.size
        val ONetIn = FloatArray(num * 48 * 48 * 3)
        val curCrop = FloatArray(48 * 48 * 3)
        var ONetInIdx = 0
        for (i in 0 until num) {
            crop_and_resize(bitmap, boxes[i], 48, curCrop)
            MtcnnUtils.flip_diag(curCrop, 48, 48, 3)
            for (j in curCrop.indices) ONetIn[ONetInIdx++] = curCrop[j]
        }
        //Run ONet
        ONetForward(ONetIn, boxes)
        //ONetThreshold
        for (i in 0 until num)
            if (boxes[i].score < ONetThreshold)
                boxes[i].deleted = true
        BoundingBoxReggression(boxes)
        //Nms
        nms(boxes, 0.7f, "Min")
        return MtcnnUtils.updateBoxes(boxes)
    }

    private fun square_limit(boxes: Vector<Box>, w: Int, h: Int) {
        //square
        for (i in boxes.indices) {
            boxes[i].toSquareShape()
            boxes[i].limit_square(w, h)
        }
    }

    /*
     * 参数：
     *   bitmap:要处理的图片
     *   minFaceSize:最小的人脸像素值.(此值越大，检测越快)
     * 返回：
     *   人脸框
     */
    fun detectFaces(bitmap: Bitmap, minFaceSize: Int): Vector<Box> {
        val t_start = System.currentTimeMillis()
        //【1】PNet generate candidate boxes
        var boxes = PNet(bitmap, minFaceSize)
        square_limit(boxes, bitmap.width, bitmap.height)
        //【2】RNet
        boxes = RNet(bitmap, boxes)
        square_limit(boxes, bitmap.width, bitmap.height)
        //【3】ONet
        boxes = ONet(bitmap, boxes)
        //return
        Log.i(TAG, "[*]Mtcnn Detection Time:" + (System.currentTimeMillis() - t_start))
        lastProcessTime = System.currentTimeMillis() - t_start
        return boxes
    }


}