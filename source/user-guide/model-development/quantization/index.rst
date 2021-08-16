.. _quantization-guide:

量化
^^^^

常见的深度学习网络，其参数和运算的数据类型一般是浮点类型，而工业界出于对特定场景的速度和内存需求，需要把模型转换为基于定点运算和存储的形式。一般把这种形式称为量化。
量化的原因有两点：
* 某些场景需要更小的内存和更快的推理速度，量化一般都是将32bit的浮点数转换成8bit,4bit等定点数，首先在内存占用上就有4x以上的缩减，同时具有更少的运行时内存和缓存要求，此外因为大部分的硬件对于定点运算都有特定的优化，所以在运行速度上也会有较大的提升
* 某些计算设备只支持定点运算，这种情况下为了让网络可以在这些设备上正常运行，我们需要对其进行量化处理
“当然，以上两点成立的前提模型经过量化处理之后，模型的正确率并没有下降或者下降后仍处于可用的状态。幸运的是，大部分情况下，量化都能得到相对不错的结果(此处应有白皮书链接)”

MegEngine针对量化提供了一整套的方案，节省了大量我们需要手动处理的时间。不过为了理解它的工作，我们需要对量化的原理和实现有一个初步的了解。
接下来我们将介绍一些量化的基础知识，有相关背景的同学可以直接浏览下面的实例讲解部分。

量化入门
-----------

前面提到了量化就是将基于浮点数据类型的模型转换为定点进行运算，其核心就是如何用定点数去表示模型中的浮点数，以及如何用定点运算去表示对应的浮点运算。
以量化常见的float32转uint8为例，一种最简单的转换方法是直接舍去float32的小数部分，只取其中的整数部分，并且对于超出(0,255)表示范围的值用0或者255表示。
这种方案显然是不合适的，尤其是深度神经网络经过bn处理后，其中间层输出基本都是0均值，1方差的数据范围，在这种方案下因为小数部分被舍弃掉了，会带来大量的
精度损失。并且因为(0,255)以外的部分被clip到了0或255，当浮点数为负或者大于255的情况下，会导致巨大的误差。

这个方案比较接近我们常见编程语言中的类型转换逻辑，我们可以把它称之为类型转换方案。上面的分析可以看出，类型转换方案对于过大或者过小的数据都会产生较大的精度损失。

目前主流的浮点转定点方案基本采用均匀量化，因为这种方案对推理更友好。将一个浮点数根据其值域范围，均匀的映射到一个定点数的表达范围上。

均匀量化方案
~~~~~~~~~~~~~
我们假设一个浮点数x的值域范围为{x_min, x_max}，要转换到一个表达范围为(0,255)的8bit定点数的转换公式如下
\begin{gather*}
x_{int} = round(x/s) + z
x_{Q} = clamp(0,255,x_{int}) 
\end{gather*}
其中s为scale，也叫步长，是个浮点数。
.. math::
:nowrap:
\begin{gather*}
scale = (x_{max} - x_{min}) / 255
\end{gather*}
z为零点，即浮点数中的0，是个定点数。 
.. math::
:nowrap:
\begin{gather*}
z = round(0 - x_{min}) / 255
\end{gather*}

由上可以看出均匀量化方案对于任意的值域范围都能表达相对不错的性能，不会存在类型转换方案的过小值域丢精度和过大值域无法表示的情况。
代价是需要额外引入零点z和值域s两个变量。同时我们可以看出，均匀量化方案因为round和clamp操作也是存在精度损失的，所以会对模型的性能产生影响。
如何减轻数据从浮点转换到定点的精度损失，是整个量化研究的重点。


“注意零点很重要，因为我们的网络模型的padding，relu等运算对于0比较敏感，需要被正确量化才能保证转换后的定点运算的正确性。当浮点数的值域范围不包含零点的时候，为了保证正确量化，我们需要对其值域范围进行一定程度的缩放使其可以包含0点”

均匀量化方案对应的反量化公式如下
.. math::
:nowrap:
\begin{gather*}
x_{float} = (x_{Q} - z) * s
\end{gather*}

所以经过量化和反量化之后的浮点数与原来的浮点数存在一定的误差，这个过程的差异可以查看下图。量化对我们网络模型的参数进行了离散化，这种操作对于模型最终点数的影响程度取决于我们模型本身的参数分布与均匀分布的差异
此处需要插入图片，

接下来我们来看看如何用经过量化运算的定点卷积运算去表示一个原始的浮点卷积操作

.. math::
:nowrap:
\begin{gather*}
conv(x, w)  = conv((x_{Q} - z_{x}) * s_{x}, (w_{Q} - z_{w}) * s_{w}) \\conv(x, w) = s_{x}s_{w} conv(x_{Q} - z_{x},w_{Q} - z_{w} ) \\conv(x, w) = s_{x}s_{w} (conv(x_{Q}, w_{Q}) - z_{x} \sum_{k,l,m}x_{Q} - z_{w}\sum_{k,l,m,n}w_{Q} + z_{x}z_{w})
\end{gather*}
其中k,l,m,n分别是kernel size，output channel和input channel的遍历下标。可以看出，当卷积的输入和参数的zero_point都是0的时候，浮点卷积将简化成
.. math::
:nowrap:
\begin{gather*}
conv(x, w) = s_{x}s_{w} (conv(x_{Q}, w_{Q}))
\end{gather*}
即定点的卷积运算结果和实际输出只有一个scale上的偏差，大大的简化了定点的运算逻辑，
所以大部分情况下我们都是使用对称均匀量化。

当我们把定点量化对应的zero point固定在整型的0处时，便是对称均匀量化。我们以int8的定点数为例 (选取int8只是为了看上去更对称一些，选取uint8也是可以的), 量化公式如下

.. math::
:nowrap:
\begin{gather*}
scale = max(abs(x_{min}), abs(x_{max})) / 127
x_{int} = round(x/s)
x_{Q} = clamp(-128,127,x_{int})
\end{gather*}
处于更快的SIMD实现的目的，有时候我们会把卷积的weight的定点范围表示成(-127,127)，对应的反量化操作为
.. math::
:nowrap:
\begin{gather*}
x_{float} = x_{Q}*s
\end{gather*}
由此可见，对称均匀量化的量化和反量化操作会更加的便捷一些
除此之外还有随机均匀量化等别的量化手段，因为大部分情况下我们都采用对称均匀量化，这里不再展开描述。

值域统计
~~~~~~~~
上面均匀量化介绍里的关键就是scale和零点，而它们是通过浮点数的值域范围来确定的。我们如何确定网络中每个需要量化的数据
的值域范围呢，一般有以下两种方案:
* 一种是根据经验手动设定值域范围，在缺乏数据的时候或者对于一些中间feature我们可以这样做
* 还有一种是跑一批少量数据，根据统计量来进行设定，这里统计方式可以视数据特性而定。

量化感知训练
~~~~~~~~~~~
在均匀量化的小节我们提到量化前后的误差主要取决于模型的参数和激活值分布与均匀分布的差异。对于量化友好的模型，我们只需要通过
值域统计得到其值域范围，然后调用对应的量化方案进行定点化就可以了。但是对于量化不友好的模型，直接进行量化会因为误差较大而使得
最后模型的正确率过低而无法使用。有没有一种方法可以在训练的时候就提升模型对量化的友好度呢？

答案是有的，我们可以通过在训练过程中，给待量化参数进行量化和反量化的操作，便可以引入量化带来的精度损失，然后通过训练让网络逐渐
适应这种干扰，从而使得网络在真正量化后的表现与训练表现一致。这个操作就叫量化感知训练，也叫qat (Quantization-aware-training)

其中需要注意的是，因为量化操作不可导，所以在实际训练的时候做了一步近似，把上一层的导数直接跳过量化反量化操作传递给了当前参数。

网络量化
~~~~~~~~~~~~~~~
上面讲述了定点情况下卷积操作的形式，大家可以自己推导一下定点情况下激活函数relu情况。
对于bn，因为大部分网络在都会进行吸bn的操作，所以我们可以把它集成进conv里。

对于现成网络，我们可以在每个卷积层前后加上量化与反量化的操作，这样就实现了用定点运算替代浮点运算的目的。
更进一步的，我们可以在整个网络推理过程中维护每个量化变量对应的scale变量，这样我们可以在不进行反量化的情况下走完
整个网络，这样我们除了带来极少量额外的scale计算开销外，便可以将整个网络的浮点运算转换成对应的定点运算。

值域统计和量化感知训练需要涉及的操作大部分都发生在训练阶段，megengine对于这两个操作都提供了相应的封装，并不需要我们手动实现

至此我们粗略的介绍了整个网络量化的定点转换以及转换后的计算方案。

参考文献：
https://arxiv.org/pdf/1806.08342.pdf

工程实现
~~~~~~~~

一般在浮点模型到定点模型这一步中间还有一步训练步骤，但我们把这一步放到后面再讲。我们这一节主要讲一下megengine是如何完成量化转化的，以及在实际运行过程中是怎么一回事。
为了方便批量操作，megengine 把module整理成了三类

* 进行浮点运算的 默认 Module
* 为qat使用的带有伪量化算子和observe算子的 QATModule
* 最终量化转化完毕的量化算子 QuantizedModule
  
对于其中比较常见的可以被量化的算子(conv等)，在这三种module中分别有同名的实现，megengine提供了quantize_qat 和 quantize 两个来完成批量的op替换操作

* quantize_qat 会把float module 转换成qat_module，通过 qat_module的源码 我们可以看出
  * 在转换过程中qat_module本身根据qconfig相关配置设置对应module的weight (权重)和act (激活值)的 observe和fake_quant
  * 在之后qat_module的forward过程中，qat_module会在调用 _apply_fakequant_with_observer 的时候对相应的tensor进行统计值域和进行伪量化的操作
* quantize 主要是将一个qat_module转换成真正的quantized_module，在这一步会执行上面提到的浮点转定点操作，根据qat_module统计的观测值和设置的定点类型将qat_module里的weight转换成对应的定点类型

所以在megengine上做一个常规的量化流程：
#. 首先将包含Module的常规模型转换成带qat_module的模型，这一步需要配置Qconfig，然后调用 quantize_qat 将module中可被量化的算子转换成同名的qat算子
#. 如果需要进行qat训练，我们在第一步配置qconfig的时候需要指定伪量化算子，然后进行训练。同时每个对应qat算子的observe会统计需要量化的tensor的值域范围。
   #. 如果只是进行calibration，只需要把伪量化算子置为None即可
#. 调用quantize将qat_module转换成quantize_module，这一步将进行实际的浮点转量化操作

接口介绍
--------

在 MegEngine 中，最上层的接口是配置如何量化的 :class:`~.quantization.QConfig` 
和模型转换模块里的 :func:`~.quantization.quantize_qat` 与 :func:`~.quantization.quantize` 。

QConfig
~~~~~~~

QConfig 包括了 :class:`~.quantization.Observer` 和 :class:`~.quantization.FakeQuantize` 两部分。
我们知道，对模型转换为低比特量化模型一般分为两步：
一是统计待量化模型中参数和 activation 的数值范围（scale）和零点（zero_point），
二是根据 scale 和 zero_point 将模型转换成指定的数值类型。而为了统计这两个值，我们需要使用 Observer.

Observer 继承自 :class:`~.module.Module` ，也会参与网络的前向传播，
但是其 forward 的返回值就是输入，所以不会影响网络的反向梯度传播。
其作用就是在前向时拿到输入的值，并统计其数值范围，并通过 :meth:`~.quantization.Observer.get_qparams` 来获取。
所以在搭建网络时把需要统计数值范围的的 Tensor 作为 Observer 的输入即可。

.. code-block::

    # forward of MinMaxObserver
    def forward(self, x_orig):
        if self.enabled:
            # stop gradient
            x = x_orig.detach()
            # find max and min
            self.min_val._reset(F.minimum(self.min_val, x.min()))
            self.max_val._reset(F.maximum(self.max_val, x.max()))
        return x_orig

另外如果只观察而不模拟量化会导致模型掉点，于是我们需要有 FakeQuantize 
来根据 Observer 观察到的数值范围模拟量化时的截断，使得参数在训练时就能提前“适应“这种操作。
FakeQuantize 在前向时会根据传入的 scale 和 zero_point 对输入 Tensor 做模拟量化的操作，
即先做一遍数值转换再转换后的值还原成原类型，如下所示：

.. code-block::

    def fake_quant_tensor(inp: Tensor, qmin: int, qmax: int, q_dict: Dict) -> Tensor:
        scale = q_dict["scale"]
        zero_point = 0
        if q_dict["mode"] == QuantMode.ASYMMERTIC:
            zero_point = q_dict["zero_point"]
        # Quant
        oup = Round()(inp / scale) + zero_point
        # Clip
        oup = F.minimum(F.maximum(oup, qmin), qmax)
        # Dequant
        oup = (oup - zero_point) * scale
        return oup

目前 MegEngine 支持对 weight/activation 两部分的量化，如下所示：

.. code-block::

    ema_fakequant_qconfig = QConfig(
        weight_observer=partial(MinMaxObserver, dtype="qint8", narrow_range=True),
        act_observer=partial(ExponentialMovingAverageObserver, dtype="qint8", narrow_range=False),
        weight_fake_quant=partial(FakeQuantize, dtype="qint8", narrow_range=True),
        act_fake_quant=partial(FakeQuantize, dtype="qint8", narrow_range=False),
    )

这里使用了两种 Observer 来统计信息，而 FakeQuantize 使用了默认的算子。

如果是后量化，或者说 Calibration，由于无需进行 FakeQuantize，故而其 fake_quant 属性为 None 即可：

.. code-block::

    calibration_qconfig = QConfig(
        weight_observer=partial(MinMaxObserver, dtype="qint8", narrow_range=True),
        act_observer=partial(HistogramObserver, dtype="qint8", narrow_range=False),
        weight_fake_quant=None,
        act_fake_quant=None,
    )

除了使用在 :class:`~.quantization.Qconfig` 里提供的预设 QConfig，
也可以根据需要灵活选择 Observer 和 FakeQuantize  实现自己的 QConfig。目前提供的 Observer 包括：

* :class:`~.quantization.MinMaxObserver` ，
  使用最简单的算法统计 min/max，对见到的每批数据取 min/max 跟当前存的值比较并替换，
  基于 min/max 得到 scale 和 zero_point；
* :class:`~.quantization.ExponentialMovingAverageObserver` ，
  引入动量的概念，对每批数据的 min/max 与现有 min/max 的加权和跟现有值比较；
* :class:`~.quantization.HistogramObserver` ，
  更加复杂的基于直方图分布的 min/max 统计算法，且在 forward 时持续更新该分布，
  并根据该分布计算得到 scale 和 zero_point。

对于 FakeQuantize，目前还提供了 :class:`~.quantization.TQT` 算子，
另外还可以继承 ``_FakeQuant`` 基类实现自定义的假量化算子。

在实际使用过程中，可能需要在训练时让 Observer 统计并更新参数，但是在推理时则停止更新。
Observer 和 FakeQuantize 都支持 :meth:`~.quantization.Observer.enable` 
和 :meth:`~.quantization.Observer.disable` 功能，
且 Observer 会在 :meth:`~module.Module.train` 
和 :meth:`~module.Module.eval` 时自动分别调用 enable/disable。

所以一般在 Calibration 时，会先执行 ``net.eval()`` 保证网络的参数不被更新，
然后再执行 :``enable_observer(net)`` 来手动开启 Observer 的统计修改功能。

模型转换模块与相关基类
~~~~~~~~~~~~~~~~~~~~~~

QConfig 提供了一系列如何对模型做量化的接口，而要使用这些接口，
需要网络的 Module 能够在 forward 时给参数、activation 加上 Observer 和进行 FakeQuantize.
转换模块的作用就是将模型中的普通 Module 替换为支持这一系列操作的 :class:`~.module.qat.QATModule` ，
并能支持进一步替换成无法训练、专用于部署的 :class:`~.module.quantized.QuantizedModule` 。

基于三种基类实现的 Module 是一一对应的关系，通过转换接口可以依次替换为不同实现的同名 Module。
同时考虑到量化与算子融合（Fuse）的高度关联，我们提供了一系列预先融合好的 Module，
比如 :class:`~.module.ConvRelu2d` 、 :class:`~.module.ConvBn2d` 和 :class:`~.module.ConvBnRelu2d` 等。
除此之外还提供专用于量化的 :class:`~.module.QuantStub` 、 :class:`~.module.DequantStub` 等辅助模块。

转换的原理很简单，就是将父 Module 中可被量化（Quantable）的子 Module 替换为对应的新 Module. 
但是有一些 Quantable Module 还包含 Quantable 子 Module，比如 ConvBn 就包含一个 Conv2d 和一个 BatchNorm2d，
转换过程并不会对这些子 Module 进一步转换，原因是父 Module 被替换之后，
其 forward 计算过程已经完全不同了，不会再依赖于这些子 Module。

.. note::

    如果需要使一部分 Module 及其子 Module 保留 Float 状态，不进行转换，
    可以使用 :meth:`~.module.Module.disable_quantize` 来处理。

    如果网络结构中涉及一些二元及以上的 ElementWise 操作符，比如加法乘法等，
    由于多个输入各自的 scale 并不一致，必须使用量化专用的算子，并指定好输出的 scale. 
    实际使用中只需要把这些操作替换为 :class:`~.module.Elemwise` 即可，
    比如 ``self.add_relu = Elemwise("FUSE_ADD_RELU")``

    另外由于转换过程修改了原网络结构，模型保存与加载无法直接适用于转换后的网络，
    读取新网络保存的参数时，需要先调用转换接口得到转换后的网络，才能用 load_state_dict 将参数进行加载。

实例讲解
--------

下面我们以 ResNet18 为例来讲解量化的完整流程，完整代码见 `MegEngine/Models <https://github.com/MegEngine/Models/tree/master/official/quantization>`_ . 主要分为以下几步：

1. 修改网络结构，使用已经 Fuse 好的 ConvBn2d、ConvBnRelu2d、ElementWise 代替原先的 Module；
2. 在正常模式下预训练模型，并在每轮迭代保存网络检查点；
3. 调用 :func:`~.quantization.quantize_qat` 转换模型，并进行 finetune；
4. 调用 :func:`~.quantization.quantize` 转换为量化模型，并执行 dump 用于后续模型部署。

网络结构见 ``resnet.py`` ，相比惯常写法，我们修改了其中一些子 Module，
将原先单独的 ``conv``, ``bn``, ``relu`` 替换为 Fuse 过的 Quantable Module。

.. code-block::

    class BasicBlock(Module):
        def __init__(self, in_planes, planes, stride=1):
            super(BasicBlock, self).__init__()
            self.conv_bn_relu = ConvBnRelu2d(
                in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False
            )
            self.conv_bn = ConvBn2d(
                planes, planes, kernel_size=3, stride=1, padding=1, bias=False
            )
            self.add_relu = Elemwise("FUSE_ADD_RELU")
            self.shortcut = Sequential()
            if stride != 1 or in_planes != planes:
                self.shortcut = Sequential(
                    ConvBn2d(in_planes, planes, kernel_size=1, stride=stride, bias=False)
                )

        def forward(self, x):
            out = self.conv_bn_relu(x)
            out = self.conv_bn(out)
            cut = self.shortcut(x)
            out = self.add_relu(out, cut)
            return out

然后对该模型进行若干轮迭代训练，并保存检查点，这里省略细节：

.. code-block::

    for step in range(0, total_steps):
        # Linear learning rate decay
        epoch = step // steps_per_epoch
        learning_rate = adjust_learning_rate(step, epoch)

        image, label = next(train_queue)
        image = tensor(image.astype("float32"))
        label = tensor(label.astype("int32"))

        n = image.shape[0]

        loss, acc1, acc5 = train_func(image, label, net, gm)
        optimizer.step()
        optimizer.clear_grad()

再调用 :func:`~.quantization.quantize_qat` 来将网络转换为 QATModule：

.. code-block::

    from ~.quantization import ema_fakequant_qconfig
    from ~.quantization.quantize import quantize_qat

    model = ResNet18()
    if args.mode != "normal":
        quantize_qat(model, ema_fakequant_qconfig)

这里使用默认的 ``ema_fakequant_qconfig`` 来进行 ``int8`` 量化。

然后我们继续使用上面相同的代码进行 finetune 训练。
值得注意的是，如果这两步全在一次程序运行中执行，那么训练的 trace 函数需要用不一样的，
因为模型的参数变化了，需要重新进行编译。
示例代码中则是采用在新的执行中读取检查点重新编译的方法。

在 QAT 模式训练完成后，我们继续保存检查点，执行 ``inference.py`` 并设置 ``mode`` 为 ``quantized`` ，
这里需要将原始 Float 模型转换为 QAT 模型之后再加载检查点。

.. code-block::

    from ~.quantization.quantize import quantize_qat
    model = ResNet18()
    if args.mode != "normal":
        quantize_qat(model, ema_fakequant_qconfig)
    if args.checkpoint:
        logger.info("Load pretrained weights from %s", args.checkpoint)
        ckpt = mge.load(args.checkpoint)
        ckpt = ckpt["state_dict"] if "state_dict" in ckpt else ckpt
        model.load_state_dict(ckpt, strict=False)

模型转换为量化模型包括以下几步：

.. code-block::

    from ~.quantization.quantize import quantize

    # 定义trace函数，打开capture_as_const以进行dump
    @jit.trace(capture_as_const=True)
    def infer_func(processed_img):
        model.eval()
        logits = model(processed_img)
        probs = F.softmax(logits)
        return probs

    # 执行模型转换
    if args.mode == "quantized":
        quantize(model)

    # 准备数据
    processed_img = transform.apply(image)[np.newaxis, :]
    if args.mode == "normal":
        processed_img = processed_img.astype("float32")
    elif args.mode == "quantized":
        processed_img = processed_img.astype("int8")

    # 执行一遍evaluation
    probs = infer_func(processed_img)

    # 将模型 dump 导出
    infer_func.dump(output_file, arg_names=["data"])

至此便得到了一个可用于部署的量化模型。
