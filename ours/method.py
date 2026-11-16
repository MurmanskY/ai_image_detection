from baselines.base import METHODS
from baselines.cnn_detection import CNNDetection


@METHODS.register("ours")
class OurMethod(CNNDetection):
    """我们的方法：面向自回归 (AR) 生成图像的鉴别。

    动机: AR 模型（LlamaGen、VAR、DALL·E 3、GPT-4o 图像生成等）将图像编码为
    离散 token 再逐个/逐尺度生成，其伪影模式与扩散模型/GAN 不同
    （例如 VQ 码本带来的网格状伪影、跨 patch 全局不一致性）。

    当前继承 CNNDetection 作为起点，可重写的扩展点:
      - get_transform(train): 自定义预处理/增广
      - build_model(): 自定义网络结构（多分支、频域+空域融合等）
    """

    def build_model(self):
        # TODO: 替换为我们方法的网络结构
        return super().build_model()
