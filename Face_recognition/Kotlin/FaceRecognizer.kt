import android.graphics.Bitmap
import org.tensorflow.lite.DataType
import org.tensorflow.lite.support.image.TensorImage
import timber.log.Timber
import java.util.*
import javax.inject.Inject
import kotlin.math.pow

class FaceRecognizer @Inject constructor(private val faceRecognitionModel: Facenet) {

    private var clusterTable = Hashtable<Long, Cluster>()
    private var lastCleaning = System.currentTimeMillis()
    private var nameSet = mutableSetOf(
            "Almond",
            "Apple",
            "Apricot",
            "Bagel",
            "Basil",
            "Bean",
            "Biscuit",
            "Boba
            "Brownie",
            "Butter",
            "Butternut",
            "Butterscotch",
            "Cannoli",
            "Cappuccino",
            "Caramel",
            "Cheesecake",
            "Chip",
            "Chocolate",
            "Chocolate Chip",
            "Churro",
            "Cinnamon",
            "Clementine",
            "Coco",
            "Cocoa",
            "Coconut",
            "Coffee",
            "Cookie",
            "Crumpet",
            "Cupcake",
            "Donut",
            "Dot",
            "Dumpling",
            "Espresso",
            "Figgy",
            "Frappe",
            "Frito",
            "Fudge",
            "Granola",
            "Gummi",
            "Hazelnut",
            "Jelly",
            "Jellybean",
            "Jujube",
            "Kimchi",
            "KitKat",
            "Kiwi",
            "Latte",
            "Lychee",
            "Macaron",
            "Mango",
            "Marshmallow",
            "Matcha",
            "Meringue",
            "Mint",
            "Mocha",
            "Mochi",
            "Mousse",
            "Muffin",
            "Nacho",
            "Nougat",
            "Olive",
            "Oreo",
            "Pancake",
            "Peaches",
            "Peanut",
            "Pecan",
            "Pepper",
            "Pistachio",
            "Pretzel",
            "Pudding",
            "Pumpkin",
            "Raisin",
            "Reeses",
            "Sesame",
            "Shortcake",
            "Smores",
            "Snickerdoodle",
            "Sprinkles",
            "Strudel",
            "Sundae",
            "Sushi",
            "Taco",
            "Taffy",
            "Tater tot",
            "Tiramisu",
            "Toffee",
            "Tofu",
            "Tootsie",
            "Truffles",
            "Twinkie",
            "Vanilla Bean",
            "Waffles",
            "Wasabi"
    )
    private var usedNameTable = Hashtable<Long, String>()

    fun recognize(faceBitmap: Bitmap, recording: Boolean): RecognizedFace {
        val faceDescriptor = getFaceNetDescriptor(faceBitmap)
        if (faceDescriptor != null) {
            return computeClusters(faceDescriptor)
        }
        return RecognizedFace()
    }

    private fun getFaceNetDescriptor(normalizedImage: Bitmap): FloatArray? {
        val inputFeature = TensorImage(DataType.FLOAT32)
        if (normalizedImage != null) {
            inputFeature.load(normalizedImage)

            // Runs model inference and gets result.
            val outputs = faceRecognitionModel.process(inputFeature.tensorBuffer)
            outputs?.let {
                val descriptor = outputs.outputFeature0AsTensorBuffer

                var magnitude = 0.0
                for (element in descriptor.floatArray) {
                    magnitude += element.toDouble().pow(2.0)
                }
                magnitude = magnitude.pow(0.5)
                val result = FloatArray(descriptor.floatArray.size)
                for (i in descriptor.floatArray.indices) {
                    result[i] = (descriptor.floatArray[i] / magnitude).toFloat()
                }
                return result
            }
        }
        return null
    }

    private fun computeClusters(faceDescriptor: FloatArray): RecognizedFace {

        // Generate clusters.
        val cluster = Cluster(System.currentTimeMillis(), arrayOf(faceDescriptor), 1, false)
        clusterTable[cluster.clusterId] = cluster
        val similarClusters = Hashtable<Long, Cluster>()
        clusterTable.keys().iterator().forEach { key ->
            if (clusterTable[key]?.isSimilar(faceDescriptor)!!) {
                similarClusters[key] = clusterTable[key]
            }
        }

        var resultIsFriend = false
        var resultFriendID = -1L
        var resultCoreFaces: Array<FloatArray>? = null
        // Merge clusters.
        if (similarClusters.size > 0) {
            val mergedCluster = mergeClusters(similarClusters)
            similarClusters.keys().iterator().forEach {
                clusterTable.remove(it)
                if (it != mergedCluster?.clusterId) {
                    releaseName(it)
                }
            }
            mergedCluster?.let {
                clusterTable[mergedCluster.clusterId] = mergedCluster
                resultIsFriend = mergedCluster.isFriend
                resultFriendID = mergedCluster.clusterId
                resultCoreFaces = mergedCluster.coreFaces
            }
        }

        // clean clusterTable per hour
        if (System.currentTimeMillis() - lastCleaning > 3600000) {
            cleanClusterTable()
            lastCleaning = System.currentTimeMillis()
        }

        // Log detected clusters.
        if (BuildConfig.DEBUG) {
            Timber.d("clusters table:")
            clusterTable.keys().iterator().forEach { key ->
                clusterTable[key]?.let { cluster ->
                    Timber.d("Cluster detected: id=%s, isFriend=%s, count=%d", cluster.clusterId, cluster.isFriend.toString(), cluster.count)
                }
            }
            Timber.d(resultFriendID.toString() + "; "+ resultIsFriend + "; "+ usedNameTable[resultFriendID])
        }

        return if (resultIsFriend) {
            RecognizedFace(resultFriendID.toString(), resultIsFriend, resultCoreFaces, assignName(resultFriendID))
        }
        else {
            RecognizedFace(resultFriendID.toString(), resultIsFriend, resultCoreFaces, "NULL")
        }
    }

    private fun mergeClusters(cs: Hashtable<Long, Cluster>): Cluster? {
        val mergedCluster = cs.values.first()
        var mergedClusterID = mergedCluster.clusterId
        var largestClusterCount = mergedCluster.count
        cs.values.filterIndexed { index, _ -> index != 0 }.forEach { c ->
            mergedCluster?.extend(c)
            if (c.count > largestClusterCount) {
                mergedClusterID = c.clusterId
                largestClusterCount = c.count
            }
            else if (c.clusterId < mergedClusterID && c.isFriend) {
                mergedClusterID = c.clusterId
            }
        }
        mergedCluster.clusterId = mergedClusterID
        return mergedCluster
    }

    fun stop() {
        faceRecognitionModel?.close()
    }

    private fun assignName(id: Long): String {
        return if (usedNameTable[id] != null) {
            usedNameTable[id]!!
        }
        else {
            val nameToAssign = nameSet.random()
            nameSet.removeIf {
                it == nameToAssign
            }
            usedNameTable[id] = nameToAssign
            nameToAssign
        }
    }
    private fun releaseName(id: Long) {
        if (usedNameTable[id] != null) {
            nameSet.add(usedNameTable[id]!!)
            usedNameTable.remove(id)
        }
    }
    private fun changeName(id: Long, nameToSet: String) {
        if (usedNameTable[id] != null) {
            nameSet.add(usedNameTable[id]!!)
            usedNameTable[id] = nameToSet
        }
    }

    private fun cleanClusterTable() {
        clusterTable.keys().iterator().forEach { key ->
            if (clusterTable[key]?.isFriend == false) {
                clusterTable.remove(key)
            }
        }
    }

    class Cluster(var clusterId: Long, var coreFaces: Array<FloatArray>, var count: Int, var firstSeen: Boolean) {

        var isFriend = false

        private fun calcDistance(face1: FloatArray, face2: FloatArray): Double {
            val l = face1.size
            var norm = 0.0
            for (i in 0 until l) {
                norm += (face1[i] - face2[i]).toDouble().pow(2.0)
            }
            return norm.pow(0.5)
        }

        fun isSimilar(face: FloatArray): Boolean {
            var result = false
            for (cf in coreFaces)
            {
//            Timber.d(calcDistance(face, cf).toString())
                if (calcDistance(face, cf) < Constants.CLUSTER_SIMILARITY_THRESHOLD) {
                    result = true
                }
            }
            return result
        }

        fun extend(cluster: Cluster) {
            coreFaces += cluster.coreFaces
            count += cluster.count
            isFriend = count > Constants.FRIEND_THRESHOLD
        }
    }

    data class RecognizedFace constructor(
        var id: String = "NULL",
        var isFriend: Boolean = false,
        var coreFaces: Array<FloatArray>? = null,
        var name: String = "NULL"
    ) {
        override fun equals(other: Any?): Boolean {
            if (this === other) return true
            if (javaClass != other?.javaClass) return false

            other as RecognizedFace

            if (id != other.id) return false
            if (isFriend != other.isFriend) return false
            if (coreFaces != null) {
                if (other.coreFaces == null) return false
                if (!coreFaces.contentDeepEquals(other.coreFaces)) return false
            } else if (other.coreFaces != null) return false
            if (name != other.name) return false

            return true
        }

        override fun hashCode(): Int {
            var result = id.hashCode()
            result = 31 * result + isFriend.hashCode()
            result = 31 * result + (coreFaces?.contentDeepHashCode() ?: 0)
            result = 31 * result + name.hashCode()
            return result
        }
    }
}