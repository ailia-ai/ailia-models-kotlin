package jp.axinc.ailia_kotlin

import android.graphics.Canvas
import android.graphics.Color
import android.graphics.Paint
import kotlin.math.hypot
import kotlin.math.roundToInt

internal data class OrtPosePoint(val x: Float, val y: Float, val score: Float)
internal data class OrtLightweightPose(val points: List<OrtPosePoint?>)

private data class PoseCandidate(
    val id: Int,
    val type: Int,
    val x: Int,
    val y: Int,
    val score: Float,
)

private data class PoseConnection(val first: Int, val second: Int, val score: Float)

// COCO keypoint order used by the Lightweight Human Pose model.
private val POSE_LIMBS = arrayOf(
    intArrayOf(1, 2), intArrayOf(1, 5), intArrayOf(2, 3), intArrayOf(3, 4),
    intArrayOf(5, 6), intArrayOf(6, 7), intArrayOf(1, 8), intArrayOf(8, 9),
    intArrayOf(9, 10), intArrayOf(1, 11), intArrayOf(11, 12), intArrayOf(12, 13),
    intArrayOf(1, 0), intArrayOf(0, 14), intArrayOf(14, 16), intArrayOf(0, 15),
    intArrayOf(15, 17), intArrayOf(2, 16), intArrayOf(5, 17),
)

private val POSE_PAF_CHANNELS = arrayOf(
    intArrayOf(12, 13), intArrayOf(20, 21), intArrayOf(14, 15), intArrayOf(16, 17),
    intArrayOf(22, 23), intArrayOf(24, 25), intArrayOf(0, 1), intArrayOf(2, 3),
    intArrayOf(4, 5), intArrayOf(6, 7), intArrayOf(8, 9), intArrayOf(10, 11),
    intArrayOf(28, 29), intArrayOf(30, 31), intArrayOf(34, 35), intArrayOf(32, 33),
    intArrayOf(36, 37), intArrayOf(18, 19), intArrayOf(26, 27),
)

internal fun decodeOrtLightweightPose(
    heatmaps: FloatArray,
    pafs: FloatArray,
    width: Int,
    height: Int,
    heatmapThreshold: Float = 0.1f,
    pafThreshold: Float = 0.05f,
): List<OrtLightweightPose> {
    require(width > 2 && height > 2)
    val plane = width * height
    require(heatmaps.size >= 19 * plane)
    require(pafs.size >= 38 * plane)

    val candidatesByType = Array(18) { mutableListOf<PoseCandidate>() }
    val allCandidates = mutableListOf<PoseCandidate>()
    for (type in candidatesByType.indices) {
        val channelOffset = type * plane
        for (y in 0 until height) {
            for (x in 0 until width) {
                val score = heatmaps[channelOffset + y * width + x]
                if (score < heatmapThreshold) continue
                var isPeak = true
                for (dy in -1..1) {
                    for (dx in -1..1) {
                        if (dx == 0 && dy == 0) continue
                        val neighborX = (x + dx).coerceIn(0, width - 1)
                        val neighborY = (y + dy).coerceIn(0, height - 1)
                        if (heatmaps[channelOffset + neighborY * width + neighborX] > score) {
                            isPeak = false
                        }
                    }
                }
                if (isPeak) {
                    val candidate = PoseCandidate(allCandidates.size, type, x, y, score)
                    candidatesByType[type] += candidate
                    allCandidates += candidate
                }
            }
        }
    }
    if (allCandidates.isEmpty()) return emptyList()

    val parent = IntArray(allCandidates.size) { it }
    val connected = BooleanArray(allCandidates.size)
    fun find(value: Int): Int {
        var root = value
        while (parent[root] != root) root = parent[root]
        var node = value
        while (parent[node] != node) {
            val next = parent[node]
            parent[node] = root
            node = next
        }
        return root
    }
    fun union(first: Int, second: Int) {
        val firstRoot = find(first)
        val secondRoot = find(second)
        if (firstRoot != secondRoot) parent[secondRoot] = firstRoot
    }

    POSE_LIMBS.indices.forEach { limbIndex ->
        val firstCandidates = candidatesByType[POSE_LIMBS[limbIndex][0]]
        val secondCandidates = candidatesByType[POSE_LIMBS[limbIndex][1]]
        if (firstCandidates.isEmpty() || secondCandidates.isEmpty()) return@forEach
        val pafChannels = POSE_PAF_CHANNELS[limbIndex]
        val proposals = mutableListOf<PoseConnection>()
        firstCandidates.forEach { first ->
            secondCandidates.forEach secondCandidate@{ second ->
                val dx = (second.x - first.x).toFloat()
                val dy = (second.y - first.y).toFloat()
                val length = hypot(dx, dy)
                if (length < 1f) return@secondCandidate
                val unitX = dx / length
                val unitY = dy / length
                var positive = 0
                var alignment = 0f
                val samples = 10
                for (sample in 0 until samples) {
                    val ratio = sample.toFloat() / (samples - 1)
                    val x = (first.x + dx * ratio).roundToInt().coerceIn(0, width - 1)
                    val y = (first.y + dy * ratio).roundToInt().coerceIn(0, height - 1)
                    val offset = y * width + x
                    val value = pafs[pafChannels[0] * plane + offset] * unitX +
                        pafs[pafChannels[1] * plane + offset] * unitY
                    alignment += value
                    if (value > pafThreshold) positive++
                }
                val score = alignment / samples
                if (positive >= 8 && score > 0f) {
                    proposals += PoseConnection(first.id, second.id, score)
                }
            }
        }
        val usedFirst = mutableSetOf<Int>()
        val usedSecond = mutableSetOf<Int>()
        proposals.sortedByDescending { it.score }.forEach { connection ->
            if (connection.first !in usedFirst && connection.second !in usedSecond) {
                usedFirst += connection.first
                usedSecond += connection.second
                connected[connection.first] = true
                connected[connection.second] = true
                union(connection.first, connection.second)
            }
        }
    }

    val components = linkedMapOf<Int, MutableList<PoseCandidate>>()
    allCandidates.forEach { candidate ->
        if (connected[candidate.id]) {
            components.getOrPut(find(candidate.id)) { mutableListOf() } += candidate
        }
    }
    return components.values.mapNotNull { component ->
        val points = arrayOfNulls<OrtPosePoint>(18)
        component.forEach { candidate ->
            val current = points[candidate.type]
            if (current == null || candidate.score > current.score) {
                points[candidate.type] = OrtPosePoint(
                    candidate.x.toFloat() / (width - 1),
                    candidate.y.toFloat() / (height - 1),
                    candidate.score,
                )
            }
        }
        if (points.count { it != null } < 3) null else OrtLightweightPose(points.toList())
    }.sortedByDescending { pose -> pose.points.count { it != null } }
}

internal fun drawOrtLightweightPoses(
    poses: List<OrtLightweightPose>,
    canvas: Canvas,
    pointPaint: Paint,
    width: Int,
    height: Int,
) {
    val linePaint = Paint().apply {
        strokeWidth = 5f
        isAntiAlias = true
        strokeCap = Paint.Cap.ROUND
    }
    poses.forEach { pose ->
        POSE_LIMBS.forEachIndexed { index, limb ->
            val first = pose.points[limb[0]]
            val second = pose.points[limb[1]]
            if (first != null && second != null) {
                linePaint.color = Color.HSVToColor(floatArrayOf(index * 360f / POSE_LIMBS.size, 1f, 1f))
                canvas.drawLine(
                    first.x * width,
                    first.y * height,
                    second.x * width,
                    second.y * height,
                    linePaint,
                )
            }
        }
        pose.points.filterNotNull().forEach { point ->
            canvas.drawCircle(point.x * width, point.y * height, 8f, pointPaint)
        }
    }
}
