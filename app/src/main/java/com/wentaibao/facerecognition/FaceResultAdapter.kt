package com.wentaibao.facerecognition

import android.widget.ImageView
import com.bumptech.glide.Glide
import com.chad.library.adapter.base.BaseQuickAdapter
import com.chad.library.adapter.base.BaseViewHolder
import com.wentaibao.facerecognition.tensorflow.FeatureSearchResult


class FaceResultAdapter(resId: Int, datas: MutableList<FeatureSearchResult>) :
        BaseQuickAdapter<FeatureSearchResult, BaseViewHolder>(resId, datas) {

    override fun convert(helper: BaseViewHolder?, item: FeatureSearchResult?) {
        val iv = helper?.getView<ImageView>(R.id.ivSrc)
        Glide.with(mContext).load(item?.feature?.src).into(iv!!)
        helper?.setText(R.id.tvName, "${item?.feature?.name}")
        helper?.setText(R.id.tvSim, "相似度:${String.format("%.2f", item?.similarity)}")
        helper?.setText(R.id.tvDis, "距离:${String.format("%.2f", item?.distance)}")
    }

}