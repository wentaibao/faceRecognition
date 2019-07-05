package com.wentaibao.facerecognition

import android.content.Intent
import android.graphics.BitmapFactory
import android.graphics.drawable.Drawable
import android.os.Bundle
import androidx.appcompat.app.AppCompatActivity
import com.bumptech.glide.Glide
import com.bumptech.glide.load.DataSource
import com.bumptech.glide.load.engine.GlideException
import com.bumptech.glide.request.RequestListener
import com.bumptech.glide.request.target.Target
import com.donkingliang.imageselector.utils.ImageSelector
import com.tencent.mmkv.MMKV
import com.wentaibao.facerecognition.tensorflow.BitmapUtil
import com.wentaibao.facerecognition.tensorflow.FaceManger
import kotlinx.android.synthetic.main.activity_main.*


class MainActivity : AppCompatActivity() {
    private val REQ_CODE = 101
    private var currentImg = "http://img.szonline.net/2019/0520/20190520024106979.jpeg"
    private var currentFeature = FloatArray(0)

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_main)

        FaceManger.initModel(applicationContext)

        val rootDir = MMKV.initialize(this)
        val mmkv = MMKV.defaultMMKV()

//        val feature = FaceFeature("z", floatArrayOf(1f), "")
//        mmkv.encode("z", feature)
//        val feature2 = mmkv.decodeParcelable("z", FaceFeature::class.java)

        btLocal.setOnClickListener {
            val bm = BitmapFactory.decodeStream(resources.assets.open("YCY2.jpg"))
            val result = FaceManger.detectFaceMTCNN(bm)
            if (result.faceCount > 0) {
                Glide.with(applicationContext).load(result.faceBm).into(ivFace)
                FaceManger.getFeature(result.faces.first())?.let { feature ->
                    currentFeature = feature
                    currentImg = "YCY1.jpg"
                    println("提取特征成功:$currentImg")
                    println(currentFeature.asList())
                }
            }

//            ImageSelector.builder().useCamera(true).setSingle(true).setViewImage(true).start(this, REQ_CODE)
        }
    }

    private fun loadImg(src: String?) {
        src?.let {
            Glide.with(this).load(it).listener(object : RequestListener<Drawable> {
                override fun onLoadFailed(
                    e: GlideException?,
                    model: Any?,
                    target: Target<Drawable>?,
                    isFirstResource: Boolean
                ): Boolean {
                    return false
                }

                override fun onResourceReady(
                    resource: Drawable?,
                    model: Any?,
                    target: Target<Drawable>?,
                    dataSource: DataSource?,
                    isFirstResource: Boolean
                ): Boolean {
                    println("onResourceReady")
//                    resource?.let { drawable ->
//                        val bm = BitmapUtil.drawableToBitmap(drawable)
//                        val result = FaceManger.detectFaceMTCNN(bm)
//                        if (result.faceCount > 0) {
//                            Glide.with(applicationContext).load(result.faceBm).into(ivFace)
//                            FaceManger.getFeature(result.faces.first())?.let { feature ->
//                                currentFeature = feature
//                                currentImg = src
//                                println("提取特征成功:$src")
//                            }
//                        }
//                    }
                    return false
                }

            }).into(ivFace)
        }
    }

    override fun onActivityResult(requestCode: Int, resultCode: Int, data: Intent?) {
        super.onActivityResult(requestCode, resultCode, data)
        if (requestCode === REQ_CODE && data != null) {
            // 获取图片选择器返回的数据
            val images = data.getStringArrayListExtra(ImageSelector.SELECT_RESULT)
            println("onActivityResult:$images")

            images?.let {
                loadImg(it.first())
            }

            // 只有本次调用相机拍出来的照片，返回时才为true 当为true时，图片返回的结果有且只有一张图片。
            val isCameraImage = data.getBooleanExtra(ImageSelector.IS_CAMERA_IMAGE, false)
        }
    }
}
