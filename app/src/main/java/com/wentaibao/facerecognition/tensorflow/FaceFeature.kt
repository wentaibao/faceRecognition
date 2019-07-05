package com.wentaibao.facerecognition.tensorflow

import android.os.Parcel
import android.os.Parcelable

data class FaceFeature(var name: String, var feature: FloatArray, var src: String) : Parcelable {
    constructor(parcel: Parcel) : this(
        parcel.readString(),
        parcel.createFloatArray(),
        parcel.readString()
    )

    override fun equals(other: Any?): Boolean {
        if (this === other) return true
        if (javaClass != other?.javaClass) return false

        other as FaceFeature

        if (name != other.name) return false
        if (!feature.contentEquals(other.feature)) return false
        if (src != other.src) return false

        return true
    }

    override fun hashCode(): Int {
        var result = name.hashCode()
        result = 31 * result + feature.contentHashCode()
        result = 31 * result + src.hashCode()
        return result
    }

    override fun writeToParcel(parcel: Parcel, flags: Int) {
        parcel.writeString(name)
        parcel.writeFloatArray(feature)
        parcel.writeString(src)
    }

    override fun describeContents(): Int {
        return 0
    }

    companion object CREATOR : Parcelable.Creator<FaceFeature> {
        override fun createFromParcel(parcel: Parcel): FaceFeature {
            return FaceFeature(parcel)
        }

        override fun newArray(size: Int): Array<FaceFeature?> {
            return arrayOfNulls(size)
        }
    }


}