# LibImage
 图片的相关处理

#### LImageUtil
图片处理工具类

* showImage 显示图片
* compressImage 压缩图片,输入图片路径,输出图片路径，宽，高，如果自适应-1

```
buildscript {

    repositories {
        google()
        jcenter()
        maven { url "https://jitpack.io" }
        maven { url 'https://maven.aliyun.com/repository/public' }  // 阿里云公共仓库‌:ml-citation{ref="3,6" data="citationList"}
        maven { url 'https://maven.aliyun.com/repository/google' }  // 阿里云Google镜像‌:ml-citation{ref="4,6" data="citationList"}
        mavenCentral()
    }
    
}
allprojects {
    repositories {
        google()
        jcenter()
        mavenCentral()
        maven { url "https://jitpack.io" }
        maven { url 'https://maven.aliyun.com/repository/public' }
    }
}
```
