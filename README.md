# Exploring Deepfakes

<a href="https://www.amazon.com/Exploring-Deepfakes-hands-generative-replacement/dp/1801810699?utm_source=github&utm_medium=repository&utm_campaign=9781801810135"><img src="https://m.media-amazon.com/images/I/71EyX0Tal9L.jpg" alt="" height="256px" align="right"></a>

This is the code repository for [Exploring Deepfakes](https://www.amazon.com/Exploring-Deepfakes-hands-generative-replacement/dp/1801810699?utm_source=github&utm_medium=repository&utm_campaign=9781801810135), published by Packt.

**Deploy powerful AI techniques for face replacement and more 
with this comprehensive guide**

## What is this book about?

This book covers the following exciting features:
Gain a clear understanding of deepfakes and their creation
Understand the risks of deepfakes and how to mitigate them
Collect efficient data to create successful deepfakes
Get familiar with the deepfakes workflow and its steps
Explore the application of deepfakes methods to your own generative needs
Improve results by augmenting data and avoiding overtraining
Examine the future of deepfakes and other generative AIs
Use generative AIs to increase video content resolution

If you feel this book is for you, get your [copy](https://www.amazon.com/dp/1801810699) today!

<a href="https://www.packtpub.com/?utm_source=github&utm_medium=banner&utm_campaign=GitHubBanner"><img src="https://raw.githubusercontent.com/PacktPublishing/GitHub/master/GitHub.png" 
alt="https://www.packtpub.com/" border="5" /></a>

## Instructions and Navigations
All of the code is organized into folders. For example, Chapter05.

The code will look like the following:
```
masker = BiSeNet(n_classes=19)
if device == "cuda":
 masker.cuda()
model_path = os.path.join(".", "binaries",
 "BiSeNet.pth")
masker.load_state_dict(torch.load(model_path))
masker.eval()
desired_segments = [1, 2, 3, 4, 5, 6, 10, 12, 13]
```

**Following is what you need for this book:**
This book is for AI developers, data scientists, and anyone looking to learn more about deepfakes or techniques and technologies from Deepfakes to help them generate new image data. Working knowledge of Python programming language and basic familiarity with OpenCV, Pillow, Pytorch, or Tensorflow is recommended to get the most out of the book.

With the following software and hardware list you can run all code files present in the book (Chapter 1-9).
### Software and Hardware List
| Chapter | Software required | OS required |
| -------- | ------------------------------------ | ----------------------------------- |
| 4-8 | Python | Windows, macOS, or Linux |
| 4-8 | PyTorch | Windows, macOS, or Linux |
| 4-8 | Pillow (PIL Fork) | Windows, macOS, or Linux |
| 4-8 | Faceswap | Windows, macOS, or Linux |
| 4-8 | OpenCV | Windows, macOS, or Linux |

### Related products
* Sentiment analysis and text classification advice [[Packt]](https://www.packtpub.com/product/network-science-with-python/9781801073691?_ga=2.232279259.578661766.1678253512-2016044401.1678253512&utm_source=github&utm_medium=repository&utm_campaign=9781801073691) [[Amazon]](https://www.amazon.com/dp/1801073694)

*  [[Packt]](https://www.packtpub.com/product/graph-data-science-with-neo4j/9781804612743?_ga=2.6892278.578661766.1678253512-2016044401.1678253512&utm_source=github&utm_medium=repository&utm_campaign=9781804612743) [[Amazon]](https://www.amazon.com/dp/180461274X)

## Get to Know the Authors
**Bryan Lyon**
is a seasoned AI expert with over a decade of experience in and around the field. His background is in computational linguistics and he has worked with the cutting-edge open source deepfake software Faceswap since 2018. Currently, Bryan serves as the chief technology officer for an AI company based in California

**Matt Tora**
is a seasoned software developer with over 15 years of experience in the field. He specializes in machine learning, computer vision, and streamlining workflows. He leads the open source deepfake project Faceswap and consults with VFX studios and tech start-ups on integrating machine learning into their pipelines.



