# LibFaceCheck

Android 人脸综合检测库，基于 MediaPipe FaceLandmarker（468 点人脸网格），一次调用即可获取人脸五官状态的全部结构化数据。

## 功能概览

| 检测项 | 说明 |
|--------|------|
| 人脸检测 | 是否检测到人脸 |
| 眼睛睁闭 | 左眼 / 右眼是否睁开（基于 EAR 值） |
| 眼睛可见性 | 左眼 / 右眼是否都能看见 |
| 鼻子可见性 | 鼻子是否能看得见 |
| 嘴巴状态 | 嘴巴是否闭合（基于 MAR 值） |
| 人脸端正 | 人脸是否端正（基于鼻子-眼睛对称性） |
| 耳朵可见性 | 左耳 / 右耳是否都能看见 |

## 环境要求

- **minSdk**: 28
- **compileSdk**: 34
- **Java**: 11+

## 依赖引入

```groovy
// settings.gradle
repositories {
    maven { url "https://jitpack.io" }
}

// build.gradle
dependencies {
    implementation project(':face_check')
}
```

## 模型文件

首次使用前需将 MediaPipe FaceLandmarker 模型文件放入 `face_check/src/main/assets/models/` 目录：

- `face_landmarker_v2_with_blendshapes.task`（推荐）
- `face_landmarker.task`（备选）

模型会自动从 assets 中加载，无需手动指定路径。

## 快速使用

### 核心 API

**`FaceDetector`** — 人脸综合检测器（实现 `Closeable`，支持 try-with-resources）

| 方法 | 参数 | 返回值 | 说明 |
|------|------|--------|------|
| `FaceDetector.create(Context context)` | `context` — Android Context | `FaceDetector` | 创建检测器实例，自动加载模型。可能抛出 `Exception` |
| `detector.detect(Bitmap bitmap)` | `bitmap` — 输入图片，**不会被回收或修改** | `FaceDetectResult` | 执行人脸综合检测 |
| `detector.close()` | 无 | void | 释放模型资源 |

### 基础用法

```java
try (FaceDetector detector = FaceDetector.create(context)) {
    FaceDetectResult result = detector.detect(bitmap);
    // bitmap 不会被回收，外部可继续使用

    if (result.isFaceDetected()) {
        // 眼睛是否都睁开
        boolean eyesOpen = result.isLeftEyeOpen() && result.isRightEyeOpen();
        // 嘴巴是否闭合
        boolean mouthClosed = result.isMouthClosed();
        // 人脸是否端正
        boolean straight = result.isFaceStraight();
        // 鼻子是否可见
        boolean noseOk = result.isNoseVisible();
        // 双眼双耳是否都可见
        boolean allVisible = result.isBothEyesVisible() && result.isBothEarsVisible();
    }
}
```

### 多次检测复用实例

```java
FaceDetector detector = FaceDetector.create(context);
try {
    // 实例可复用，无需重复加载模型
    FaceDetectResult result1 = detector.detect(bitmap1);
    FaceDetectResult result2 = detector.detect(bitmap2);
} finally {
    detector.close();
}
```

## FaceDetectResult 返回字段

### 布尔值字段

| 方法 | 返回类型 | 说明 |
|------|----------|------|
| `isFaceDetected()` | `boolean` | 是否检测到人脸 |
| `isLeftEyeOpen()` | `boolean` | 左眼是否睁开 |
| `isRightEyeOpen()` | `boolean` | 右眼是否睁开 |
| `isLeftEyeVisible()` | `boolean` | 左眼是否可见 |
| `isRightEyeVisible()` | `boolean` | 右眼是否可见 |
| `isBothEyesVisible()` | `boolean` | 双眼是否都可见 |
| `isNoseVisible()` | `boolean` | 鼻子是否可见 |
| `isMouthClosed()` | `boolean` | 嘴巴是否闭合 |
| `isFaceStraight()` | `boolean` | 人脸是否端正 |
| `isLeftEarVisible()` | `boolean` | 左耳是否可见 |
| `isRightEarVisible()` | `boolean` | 右耳是否可见 |
| `isBothEarsVisible()` | `boolean` | 双耳是否都可见 |

### 数值字段

| 方法 | 返回类型 | 说明 |
|------|----------|------|
| `getLeftEyeEAR()` | `float` | 左眼 EAR 值（Eye Aspect Ratio），> 0.18 视为睁开 |
| `getRightEyeEAR()` | `float` | 右眼 EAR 值，> 0.18 视为睁开 |
| `getMouthMAR()` | `float` | 嘴巴 MAR 值（Mouth Aspect Ratio），> 0.55 视为张开 |

### toString() 输出示例

```
FaceDetectResult{左眼=睁开(EAR=0.261), 右眼=睁开(EAR=0.245), 左眼可见=true, 右眼可见=true, 鼻子可见=true, 嘴巴=闭合(MAR=0.123), 人脸端正=true, 左耳可见=true, 右耳可见=true}
```

未检测到人脸时：

```
FaceDetectResult{未检测到人脸}
```

## 注意事项

- 传入的 **Bitmap 不会被回收或修改**，调用方可继续使用
- 传入 `null` 或已回收的 Bitmap 会返回 `isFaceDetected() = false` 的默认结果
- `FaceDetector` 实例可复用，建议创建一次多次调用 `detect()`，避免重复加载模型
- 使用完毕需调用 `close()` 释放资源，推荐 try-with-resources 方式
- 检测需在**非 UI 线程**执行，避免阻塞主线程