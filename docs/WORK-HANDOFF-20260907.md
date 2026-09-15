## 2026-09-10 goal94：恢复标注训练主线，同系统分支核对召回与误检

> 2026-09-15 规则更新：本文中的旧仓库分工仅为历史记录。两个 GitHub 仓库现为平级同步目标，不再执行开发仓向稳定仓的单向晋升；当前规则以 [双仓同步与发布规则](DUAL-REPOSITORY-RELEASE-POLICY.md) 为准。

PARTIAL_EVIDENCE，目标active。用户明确要求本任务优先标注、复核、训练；不要继续启动完整视频系统实验。本轮只做30次真实图片推理/30次NMS，新训练0次。固定原v24的15图、20个共同语义类别203实例：旧PT→v24，TP147→164、FP5→10、FN56→39，P96.71%→94.25%，R72.41%→80.79%。同RoleScanner/one2many/960/FP32/conf0.25，batch1；不是完整系统或生产引擎验收。15图均被旧模型接触，不是独立测试，不能与历史one2one全类指标混比。

实看4组对照：瓶盖收益明显，空手套误报戴手套的手、裸手重复框、药匙定位和管盖混淆仍在。TP v49沿用goal90：困难枪头0/69、原验证0/8，未替换。426图8576框1552审计不变；累计72训练不变。下一步补可靠训练来源错例并双遍复核，保持验证集；91未知来源原图仍unassigned。

前轮goal93四路实验已经结束，1146.07秒exit0、partial：4组/91候选，840秒成片机位已对应移液台，600秒显示缺乏已验证第三机位。当前action_state_machine.py另有运行结束后pending_semantic_guard修补及132个测试通过，不能把该最终补丁称为已完成93整轮实测。93浏览器在登记后重启本任务8011查看服务，尚未在重启后复核；不要把可访问入口当播放验收。93运行证据保留于system-goal93，未完成综合回执；按用户优先级先继续模型数据工作。

[本轮指标及实图](/srv/sentinel-data/VisionCortex3090Ti/Runtime/Project-Detector-Pilot-20260907/annotation-goal94/report.md) · [核验](/srv/sentinel-data/VisionCortex3090Ti/Runtime/Project-Detector-Pilot-20260907/annotation-goal94/verification-summary.json)。两仓原分支/SHA和未提交工作保留。

## 2026-09-10 goal92：补扫新增4909帧，真实机位质量仍未过门

PARTIAL_EVIDENCE，目标active。4回归修复前失败，33测试及静态检查通过；同一最终账本重放找回150/334个未充分覆盖候选。真实四路原片980.56秒结束partial，新增2轮补扫，总10517批/42053帧，2实验/86候选（29优先）/264MP4/258JPG。没有新训练，仍FP v24/TP v49，累计72次。

实看10个PTS/像素身份验证源帧及3拼图：600秒新扫c973为空移液台，称量rk仍差19.05秒，内部却判完整；需限制支持事件必须在候选自身时间内。第二段延长、实际同步422.667秒，但840秒右侧仍是空称量台，固定整段机位待修复。浏览器预览与第二段播放通过；两轮统一8011入口，列表疑似重复投影待查。待语义高级事件压制接触证据也待修复。

426图8576框1552审计未变，13受保护资产与8源文件复核通过；未替换生产，无付费。完整质量、动态机位、账号恢复复跑与正式归档仍NOT_PROVEN。上一条“goal92进行中”已由本次完成记录替代，原文保存在in-progress文档与before-docs。

[报告和实图](/srv/sentinel-data/VisionCortex3090Ti/Runtime/Project-Detector-Pilot-20260907/system-goal92/report.md) · [核验](/srv/sentinel-data/VisionCortex3090Ti/Runtime/Project-Detector-Pilot-20260907/system-goal92/verification-summary.json) · [回执](/srv/sentinel-data/VisionCortex3090Ti/Runtime/Project-Detector-Pilot-20260907/goal92-work-receipt.json)。两仓原分支/SHA与未提交修改保留。

## 2026-09-10 goal91：四路原片完成本地CV运行，发现第三人称漏扫

PARTIAL_EVIDENCE，目标active。FP v24/TP v49、960 FP32 batch4、精扫20fps，原4路约15分钟输入完整引用；832.25秒结束为partial，保留2实验/86候选，264MP4/258JPG。无新训练，累计72次（FP23/TP49）；13模型/引擎与8原输入哈希通过。未调用付费语义，不作完整质量或性能达标结论。

有效partial只读登记已修复，报告篡改拒绝访问；27测试、ruff、compileall通过。浏览器实际检查长片129.67秒和首候选6.733秒解码、初始预览可见；CLI导入的页面重跑仍未证明。10源帧PTS/像素哈希复现。发现455秒长簇被零散双机位事件整体关闭，TP精扫仅39%，600秒称量处三TP均漏扫；下一步修复逐候选覆盖后重放和实跑。完整质量、尾部实验覆盖、机位动作核验、语义与正式归档仍NOT_PROVEN。

[报告](/srv/sentinel-data/VisionCortex3090Ti/Runtime/Project-Detector-Pilot-20260907/system-goal91/report.md) · [回执](/srv/sentinel-data/VisionCortex3090Ti/Runtime/Project-Detector-Pilot-20260907/goal91-work-receipt.json)。两仓分支/SHA及未提交修改保留。

## 2026-09-10 goal90：TP v49完成，用品盒误检减少，密集枪头未达标

PROGRESS / PARTIAL_EVIDENCE，目标active。新增2张用品盒裁图，实际初标、保存后重开复核、工作台核对；426图8576框1552审计，原424条记录不变。来源明确的123训练父图+492增强=615文件，老1214图像/标签文件一致。两新裁图8个增强版本实看；模糊蓝枪头保留未知，91张未明来源不纳入训练。

v48既有23类权重继续微调10轮，960 FP32 batch2 freeze10 AdamW lr0=0.00001 one2many；实际3080更新/6150访问/163.96秒。708个模型/EMA状态张量在首次更新前核对。v49对146图、两基线各新增2图，共150真实模型调用/150 NMS；旧288预测核验复用，无ROI调用。

新用品盒训练裁图TP/FP/FN从0/3/2变2/0/0；两固定开发裁图24/2/3变24/0/3，仍3漏检。公开近景113/3/3变111/3/5，有退步；前轮17件裁图保持17/17，困难69枪头仍0/69、原验证视野0/8。13组实图复核发现空孔误报、边角用品盒漏检和新增实验服重复框。未替换生产。

累计72次完成训练（FP23/TP49），FP仍v24。两仓原分支/SHA与未提交修改保留，12模型/引擎哈希通过。下一步继续密集蓝枪头可靠标注/尺度诊断，并衔接完整原片机位、失败保留与复跑、浏览器和溯源证据；当前不具备完整系统质量验收结论。

[报告与实图](/srv/sentinel-data/VisionCortex3090Ti/Runtime/Project-Detector-Pilot-20260907/annotation-goal90/report.md) · [核验](/srv/sentinel-data/VisionCortex3090Ti/Runtime/Project-Detector-Pilot-20260907/annotation-goal90/verification-summary.json) · [回执](/srv/sentinel-data/VisionCortex3090Ti/Runtime/Project-Detector-Pilot-20260907/goal90-work-receipt.json)。

## 2026-09-10 goal89：TP v48低学习率接续训练完成，灰枪头误检恢复，未晋升

PROGRESS / PARTIAL_EVIDENCE，目标active。v46已有23类权重→lr0=0.00001微调10轮，实际3030更新/6050访问/164.91秒。相同121父图+484增强=605训练文件；输入960 FP32 batch2、freeze10、one2many。首次更新前708模型/EMA张量（含12分类输出）逐项相等；99冻结参数张量最终仍不变。未再次做21→23迁移。首次运行时观察器访问未创建的epoch属性，0更新即退出；原记录保留，新目录改start_epoch后预检完成，产品源码未改。

144整图+156盒内=300实际推理，277NMS/23空ROI；v46/v47四份历史预测在同一数据/代码/环境核验后复用。新带盖训练裁图仍17/17；公开近景TP/FP/FN由v47的103/45/13恢复113/3/3（v46为114/4/2）；T000–002恢复81/88，旧训练枪头57/61。两固定开发裁图24/2/3略逊v46的25/1/2，困难69枪头仍0/69、原验证视野0/8。已实际查看11组图，盒内防护用品误报手、空孔误报、边角防护盒漏检仍在。通用整盒ROI仍退步，不启用，不替换生产。

累计71次完成训练（FP23/TP48），FPv24不变；424图8574框1546审计和分区不变。下一步补充来源可靠的不同蓝枪头构图与防护用品/白角/空孔背景，实际逐件初标与二次复核；不凭增加轮数宣称解决。独立质量和完整原片语义归档仍NOT_PROVEN。

[报告与实图](/srv/sentinel-data/VisionCortex3090Ti/Runtime/Project-Detector-Pilot-20260907/annotation-goal89/report.md) · [核验](/srv/sentinel-data/VisionCortex3090Ti/Runtime/Project-Detector-Pilot-20260907/annotation-goal89/verification-summary.json) · [回执](/srv/sentinel-data/VisionCortex3090Ti/Runtime/Project-Detector-Pilot-20260907/goal89-work-receipt.json)。两仓原分支/SHA及未提交修改保留。

## 2026-09-10 goal88：TP v47完成，带盖构图已学到，困难域与误检仍未达标

PROGRESS / PARTIAL_EVIDENCE，目标active，v47不晋升。T031新增312×354带盖裁图经过初标/保存后第二遍复核：13枪头+1盒+2枪+1手套=17件，剔除父框矩形相交产生的不可见实验服；不是新独立来源。工作台424图8574框，1546审计，96完整/258区域外/19待复核/51排除；原423记录未改，123条部分监督批次0阻断，普通全库导出639阻断。

121训练父图2980框485枪头+484增强=605训练文件，2固定完整开发裁图不变。纳入0309更正3支及其重新生成的增强；1195旧图像标签/476旧派生记录逐项保留。960 FP32 batch2、AdamW0.0001、freeze10、20轮，真实6060更新/12100访问/321.99秒。固定last.pt；累计70次（FP23/TP47），FPv24不变。

144图×3权重整图432次+盒内465次=897真实调用，851NMS/46空ROI，旧286整图基线完全复现。新裁图1/17→17/17，同源T031ROI可检13支；困难69枪头0→0、原验证视野0/8不变。公开近景114TP/4FP/2FN→103/45/13，新增瓶盖/枪头误检；固定开发裁图25/1/2→23/2/4，虽mAP50–95 80.21%→81.69%，固定阈值P/R退步。实际查看11对比+2ROI细节，未替换生产或启用整盒ROI。

下一步从较稳v46的23类完整权重低学习率继续微调，核对分类头实际载入，不再做21→23迁移，保留同一评测与旧域门槛。91来源未明图仍unassigned。无产品源码/付费/NAS/完整原片改动，独立质量和原片语义归档仍NOT_PROVEN。

[报告及实图](/srv/sentinel-data/VisionCortex3090Ti/Runtime/Project-Detector-Pilot-20260907/annotation-goal88/report.md) · [核验](/srv/sentinel-data/VisionCortex3090Ti/Runtime/Project-Detector-Pilot-20260907/annotation-goal88/verification-summary.json) · [回执](/srv/sentinel-data/VisionCortex3090Ti/Runtime/Project-Detector-Pilot-20260907/goal88-work-receipt.json)。两仓原分支/SHA和未提交修改保留。

## 2026-09-10 goal87：整盒放大诊断未过门，转向训练构图覆盖

PROGRESS / PARTIAL_EVIDENCE，目标active。143固定图×v44/v46，真实286整图+307盒内调用，593次后处理/565次NMS，28空结果跳过NMS；整图预测全部复现。模型预测盒选区域、不用标签框，保留源像素/矩阵/机位；非枪头类别完全不变。

v46困难69枪头0→0，原验证视野0/8→0/8，旧训练枪头57/61→0/61，两整机训练枪头254/259→121/259；两固定开发裁图25/1/2不变。实际查看6组同源对比，整盒带盖构图明显退步，原T031小裁图仍13/13。暂不启用此规则，继续补充合规来源的蓝枪头和带盒盖背景训练构图，再检验两条路径。不能宣称放大已解决问题。

423图8557框/1543审计未变，累计69次训练（FP23/TP46）未变。没有生产/标签/付费/NAS/完整原片修改。包装器错误要求空结果也有NMS已按逐次账本解决，未重跑或改写预测。

[报告与实图](/srv/sentinel-data/VisionCortex3090Ti/Runtime/Project-Detector-Pilot-20260907/annotation-goal87/report.md) · [核验](/srv/sentinel-data/VisionCortex3090Ti/Runtime/Project-Detector-Pilot-20260907/annotation-goal87/verification-summary.json) · [回执](/srv/sentinel-data/VisionCortex3090Ti/Runtime/Project-Detector-Pilot-20260907/goal87-work-receipt.json)。原分支及未提交工作保留。

## 2026-09-10 goal86：三处漏标已补标并复核，旧训练记录保留

PROGRESS / PARTIAL_EVIDENCE，目标active。0309左架3处漏标已通过CLI更正并保存后实际复核至revision4：120支枪头、129框、1未知区。全库423图8557框，1543审计；其他422条、原126框几何、来源与分区均不变。122条部分监督格式0阻断，全库普通导出639阻断。

仅对旧预测按新标签重新计数，v46该图122/126→125/129；权重/预测未变，其他142图结果相同，不是模型改进。原goal85数据、训练、原分数保留；累计69次训练不变，FPv24/TPv46均未因此晋升。旧域枪头、空孔/重复框、来源独立质量与完整原片系统验收仍未过门。无新训练、模型调用、付费、NAS或生产替换。

[更正报告](/srv/sentinel-data/VisionCortex3090Ti/Runtime/Project-Detector-Pilot-20260907/annotation-goal86/report.md) · [核验](/srv/sentinel-data/VisionCortex3090Ti/Runtime/Project-Detector-Pilot-20260907/annotation-goal86/verification-summary.json) · [回执](/srv/sentinel-data/VisionCortex3090Ti/Runtime/Project-Detector-Pilot-20260907/goal86-work-receipt.json)。下一步按修正版本冻结训练数据并继续旧域质量迭代。

## 2026-09-10 goal85：TP v46完成，整机训练拟合提高，旧域枪头退步并发现3处漏标

PROGRESS / PARTIAL_EVIDENCE，目标active，v46不晋升。120训练父图2960已标框+480增强=600训练文件，原2完整验证裁图不变；960 FP32 batch2、AdamW0.0001、freeze10、20轮，实际6000更新12000访问、319.82秒。保留原120导出记录、1184图像标签及472增强记录；固定epoch20 last.pt。累计69次（FP23/TP46），FP仍v24。

143图×3模型=429次真实调用/NMS，生产和v44全部143原始预测复现。新整机训练图已标枪头0/256→251/256，原困难69枪头4→0，完整验证原图枪头0/8不变、移液枪24/26→23/26；两固定完整开发裁图23/0/4→25/1/2，mAP50–95 77.97%→80.21%。提升主要为训练拟合，不能证明整体泛化达标。

实际9对比图+4密集叠框+2原像素复查，发现0309左架r2/r3/r4第9列3支漏标；原revision2训练/评测和数值保留，下一步先CLI更正、保存后再复核。不能把这三处报为模型空孔误检，也不能静默改标签重算原成绩。右架空孔、重复框及防护盒角部纸张误检仍在。

工作台423图8554框、1541审计记录本阶段不变；122条格式/来源契约0阻断、全库普通导出639阻断，格式通过不代表无视觉漏标。91未映射原图仍unassigned；未改生产资产/源码，无付费/完整原片/NAS操作。独立质量、完整原片多机位/语义/归档仍NOT_PROVEN。

[完整报告与真实对比](/srv/sentinel-data/VisionCortex3090Ti/Runtime/Project-Detector-Pilot-20260907/annotation-goal85/report.md) · [核验](/srv/sentinel-data/VisionCortex3090Ti/Runtime/Project-Detector-Pilot-20260907/annotation-goal85/verification-summary.json) · [回执](/srv/sentinel-data/VisionCortex3090Ti/Runtime/Project-Detector-Pilot-20260907/goal85-work-receipt.json)。两仓原分支/SHA及未提交修改保留。

## 2026-09-10 goal84：部分装载整机图117支枪头双遍复核，v45将六孔板孔误报为枪头

PROGRESS / PARTIAL_EVIDENCE，目标active。新增1张OT2完整机台图，逐件117支枪头+3盒+6整体容器=126框，保留1未知；已保存后重开原图复核。全库423图8554框，95完整、258区域外、19待复核、0草稿、51排除，审计1541；浏览器实际显示新图版本2及更新统计。旧422记录和121批次版本不变，新增后122条（120训练父图+2固定验证裁图）区域外资格检查0阻断，全项目完整导出仍639阻断。

生产/v44/v45对新0309与既有0176整机图各2次真实调用，共6源调用/6NMS。两图三模型均0/270已标实物匹配；v45在新图的六孔板孔口上报6个枪头，真正117支全漏。0176原始预测精确复现goal83。部分图不报告全图误检率。8项受保护资产未变，无产品代码、新训练/导出、付费或完整视频运行。

下一步以v44的freeze10配置为对照，把已复核两张完整图与空孔背景加入增强/训练v46，并核对旧域退步。当前累计68次训练（FP23/TP45），FP v24/TP v45，未晋升；独立质量、完整原片多机位、真实语义和归档仍NOT_PROVEN。91未映射原图保持unassigned。

[报告与图像](/srv/sentinel-data/VisionCortex3090Ti/Runtime/Project-Detector-Pilot-20260907/annotation-goal84/report.md) · [核验](/srv/sentinel-data/VisionCortex3090Ti/Runtime/Project-Detector-Pilot-20260907/annotation-goal84/verification-summary.json) · [回执](/srv/sentinel-data/VisionCortex3090Ti/Runtime/Project-Detector-Pilot-20260907/goal84-work-receipt.json)。两仓原分支/SHA及未提交修改保留。

## 2026-09-10 goal83：完整机台139支枪头双遍复核，6次真实对照确认覆盖缺口

PROGRESS / PARTIAL_EVIDENCE，目标active。本轮新增1张公开OT2完整机台图，逐件139枪头+2盒+3容器=144框，保存初标后实际重开复核，修正140框边界，保留废弃盒及左边器材2未知。全库422图8428框：95完整、257区域外、19待复核、0草稿、51排除；审计1538。浏览器实际显示新图版本2、139枪头及新统计。旧421记录/120批次版本不变，加入新图后121条资格检查0阻断，全项目普通导出仍637阻断；无新导出/训练。

生产/v44/v45各2图，共6真实源调用及6NMS。完整机台144已知目标均0匹配，两候选无≥0.25输出；同名已训练近景v44为52/63枪头、v45为61/63，原始预测均复现goal82。完整图约11×9像素枪头，近景约39×39，存在明显尺寸/上下文/训练覆盖差异，但未核实原始裁剪矩阵，不作纯尺寸因果消融。生产把六孔板6孔报为样品瓶；旧类11语义不同不作单支同任务准确率基线。

累计训练仍68次（FP23/TP45），FP v24/TP v45；v45不晋升，下轮继续以v44为开发对照补完整视野及空孔。8项受保护资产不变，无产品代码/付费/完整视频/NAS操作。91未映射图保留unassigned，完整原片多机位/语义/归档和独立质量仍NOT_PROVEN。

[报告与实际图像](/srv/sentinel-data/VisionCortex3090Ti/Runtime/Project-Detector-Pilot-20260907/annotation-goal83/report.md) · [核验](/srv/sentinel-data/VisionCortex3090Ti/Runtime/Project-Detector-Pilot-20260907/annotation-goal83/verification-summary.json) · [回执](/srv/sentinel-data/VisionCortex3090Ti/Runtime/Project-Detector-Pilot-20260907/goal83-work-receipt.json)。两仓原分支/SHA及未提交修改保留。

## 2026-09-10 goal82：TP v45完成冻结范围受控训练及141图对照

PROGRESS / PARTIAL_EVIDENCE，目标active，v45不晋升生产。解冻使公开训练图枪头匹配102/113增至111/113，但空孔误检增加、旧域保留退步，原困难枪头仅6/69；本轮假设未达到可用收益。 冻结前10层改前4层，可训练参数551万→977万；118父图+472增强、2固定完整开发裁图不变。960 FP32 batch2 AdamW LR0.0001、20轮、5900更新11800访问，实际351.46秒。两轮5900条批次顺序和优化器学习率记录逐条一致，固定last.pt。累计68次训练（FP23/TP45），FP仍v24。

141图×3权重=423真实源调用及423NMS；生产与v44全部141原始预测精确复现。困难图枪头4/69→6/69，完整验证视野已标枪头0/8→0/8；两张完整开发裁图正确/误检/漏检由23/0/4变为24/1/3。开发裁图和训练图不作独立质量验收。实际查看9份图像对照。

全库421图8284框/1535审计及120行训练批次不变，批次0阻断；普通全库导出仍635阻断。未改来源/分区/生产资产/源码。91未映射图继续unassigned，无新完整视频/付费/NAS操作；独立逐类门槛、完整原片多机位/语义/归档仍NOT_PROVEN。保留冻结10层的v44作为下一轮开发对照，优先补充来源合规的部分装载架/空孔及完整视野训练实例，核查小目标尺寸覆盖；不继续无依据地扫冻结参数，也不将未映射图挪入训练。

[报告和实际对比图](/srv/sentinel-data/VisionCortex3090Ti/Runtime/Project-Detector-Pilot-20260907/annotation-goal82/report.md) · [核验](/srv/sentinel-data/VisionCortex3090Ti/Runtime/Project-Detector-Pilot-20260907/annotation-goal82/verification-summary.json) · [回执](/srv/sentinel-data/VisionCortex3090Ti/Runtime/Project-Detector-Pilot-20260907/goal82-work-receipt.json)。两仓原分支/SHA及未提交工作保留。

## 2026-09-10 goal81：新增113支枪头双遍复核，TP v44已训练并完成真实对照

PROGRESS / PARTIAL_EVIDENCE，目标active，v44不晋升生产。官方OT2数据25,166,399字节ZIP实际下载并校验MD5/SHA；219张发布图片仅从train选3张入项目，逐件初标及重开原图第二遍完成113枪头+3盒体。全库421图8284框，95完整、256区域外、19待复核、0草稿、51排除；旧418记录不变，审计1535条。已实际在浏览器显示新图版本2、63支枪头及盒体。

120条训练批次检查0阻断，118训练父图+472增强=590文件，原2张完整验证裁图不变；旧116记录、456派生及1144文件逐项相同。v44沿用v43生产TP初始化、冻结10层、960 FP32 batch2 AdamW LR0.0001、20轮，实际5900更新11800访问、315.66秒。固定last.pt。累计67次（FP23/TP44），FP仍v24。普通全项目完整导出仍635阻断。

141张冻结图×3权重=423次真实源调用/NMS，生产及v43旧137图原始预测精确复现。新增训练枪头15/113→102/113，新增试管0/15→13/15；原困难枪头仅1/69→4/69，完整验证已标枪头仍0/8。原2完整验证裁图v43为24正确/1误检/3漏检，v44为23/0/4：多漏顶部瓶盖。mAP50–95 0.76992→0.77965不能掩盖召回退步或当独立质量证明。大口公开训练图仍4空孔误作枪头、1角落误作实验服。生产旧类11与新单支枪头含义不同，另列20共同类别比较。

无生产替换、完整视频、付费调用、NAS或产品源码修改。未知原始来源91图仍unassigned；完整原片/多机位/语义/归档及独立逐类质量NOT_PROVEN。下一步针对真实困难外观和原类保留核查曝光/特征冻结，继续受控迭代。

[详细报告及实际对比图](/srv/sentinel-data/VisionCortex3090Ti/Runtime/Project-Detector-Pilot-20260907/annotation-goal81/report.md) · [核验](/srv/sentinel-data/VisionCortex3090Ti/Runtime/Project-Detector-Pilot-20260907/annotation-goal81/verification-summary.json) · [回执](/srv/sentinel-data/VisionCortex3090Ti/Runtime/Project-Detector-Pilot-20260907/goal81-work-receipt.json)。两仓原分支/SHA与未提交工作保留。

## 2026-09-10 goal80：新增不同场景15根试管，真实对照暴露覆盖缺口

PROGRESS / PARTIAL_EVIDENCE，总目标active。公开Train新图1455完成实际初看与保存后重开第二遍，26框含15根试管、保留6未知，修正手/架/玻璃瓶3框。项目418图8168框：92完整、256区域外复核、19待复核、0草稿、51排除；审计1526条。实际浏览器显示版本2、26框/6未知、15试管、第三人称/train。

原生产/v43各1次真实RoleScanner调用及NMS，固定960 FP32 batch1 .25/.7/.5；两者均0/26已知匹配、0/15试管。生产0报告框，v43唯一报告框将右侧试剂瓶附近误作枪头盒。部分标注不提供整图P/R/mAP，也不是独立测试。旧417记录和116训练版本保持，资格批次117条检查0阻断，未导出或训练；普通全库导出635阻断。仍累计66次，FP v24/TP v43，生产资产不变。

另核对OT2Eye公开枪头数据CC BY 4.0登记，但Zenodo文件/API响应超时，未导入；本轮新增的是试管场景，枪头覆盖尚未补齐。91张原始来源仍未映射/unassigned。独立质量、完整原片多机位/语义/归档仍NOT_PROVEN。

[报告和实际对比图](/srv/sentinel-data/VisionCortex3090Ti/Runtime/Project-Detector-Pilot-20260907/annotation-goal80/report.md) · [核验](/srv/sentinel-data/VisionCortex3090Ti/Runtime/Project-Detector-Pilot-20260907/annotation-goal80/verification-summary.json) · [回执](/srv/sentinel-data/VisionCortex3090Ti/Runtime/Project-Detector-Pilot-20260907/goal80-work-receipt.json)。原分支/SHA与未提交修改保留。

## 2026-09-10 goal79：137图真实局部放大对照，当前方案不接入

PROGRESS / PARTIAL_EVIDENCE，总目标active。实际1478次RoleScanner模型调用和1478次NMS，原生产/v43各137整图+602局部，130旧图原始整图预测分别逐项复现。v43增加191个预测，只多匹配4个已标实例，且全部为训练来源；新增7张困难图中的69支已标枪头前后仍1/69。两张完整开发裁剪27实例，v43整图24TP/1FP/3FN，合并后24TP/2FP/3FN，精确率96%→92.31%，召回不变88.89%。这些重复开发裁剪不是独立验证。当前预设切片不接入产品。

实际查看8来源16张整图/细节，确认跨实例枪头框和重复枪身局部框。全137图已知匹配v43 3087→3091、生产2058→2074；部分标注不能计算全图误检率。候选137整图约1.074秒，额外602局部6.022秒，只是图像批次推理。417图8142框/1523审计及116旧训练记录未变，6项模型资产哈希不变；未新增训练，累计66次，FP v24/TP v43，生产不替换。91张未映射来源仍待原视频/批次对应。独立质量、完整原片多机位/语义/归档仍NOT_PROVEN。

[报告和对比图](/srv/sentinel-data/VisionCortex3090Ti/Runtime/Project-Detector-Pilot-20260907/annotation-goal79/report.md) · [核验](/srv/sentinel-data/VisionCortex3090Ti/Runtime/Project-Detector-Pilot-20260907/annotation-goal79/verification-summary.json) · [回执](/srv/sentinel-data/VisionCortex3090Ti/Runtime/Project-Detector-Pilot-20260907/goal79-work-receipt.json)。两仓原分支/SHA与既有未提交修改保留。

## 2026-09-10 goal78：两张称量侧机位补标复核，项目8142框

PROGRESS / PARTIAL_EVIDENCE，总目标active。两张原holdout图实际初看、保存、重开整图/叠框完成第二遍，新增7框，保留6处未知；一张圆盘图仍0已知框但不是负样本，另一张明确4枪+枪架/管架/枪头盒。孔位、截断器材与过曝密集区不猜数。项目417图8142框：92完整、255区域外复核、19待复核、0草稿、51排除，审计1523条；实际浏览器显示最新统计和side-b版本4、7框/5未知、暂不进入训练。

417来源/分区、415条其他标注、旧116训练版本/哈希不变；同批次当前审计检查0阻断，全项目普通导出仍633阻断。6个模型资产实际哈希不变，无新训练/导出/模型调用，累计66次、FP v24/TP v43，生产不替换。独立质量和完整原片多机位/语义/归档仍NOT_PROVEN。继续来源核定与有效困难样本，剩余val/test未知不能改状态或分区绕过。

[报告](/srv/sentinel-data/VisionCortex3090Ti/Runtime/Project-Detector-Pilot-20260907/annotation-goal78/report.md) · [核验](/srv/sentinel-data/VisionCortex3090Ti/Runtime/Project-Detector-Pilot-20260907/annotation-goal78/verification-summary.json) · [回执](/srv/sentinel-data/VisionCortex3090Ti/Runtime/Project-Detector-Pilot-20260907/goal78-work-receipt.json)。两仓原分支/SHA及未提交修改保留。

## 2026-09-10 goal77：16张原图新增439框，待初标清零

PROGRESS / PARTIAL_EVIDENCE，总目标active。实际初看16原图/33裁剪；保存后实际重开16原图/16叠框及9份细节，修正5框边界并补1处过曝未知区，再实看6份最终叠框。逐支移液枪、裸手/已戴手套手、开口/有盖试剂瓶、折纸和移位物体均按本帧可见范围核对。新标439框/34未知区。项目417图8135框：92完整、253区域外复核、21待复核、0草稿、51排除；审计1519条。浏览器T115实际显示版本2、28框/2未知及最新统计。

417来源/分区、401条其他标注、116条旧训练版本不变；同记录当前审计检查0阻断，普通全项目完整导出仍633阻断。新图继续未映射/unassigned，无新训练/导出/模型调用，累计66次，FP v24/TP v43，6个受保护模型资产哈希不变。整体优于生产与完整原片多机位/语义/归档验收仍NOT_PROVEN。继续21张待复核和来源核定，再冻结新训练批次。

[报告](/srv/sentinel-data/VisionCortex3090Ti/Runtime/Project-Detector-Pilot-20260907/annotation-goal77/report.md) · [核验](/srv/sentinel-data/VisionCortex3090Ti/Runtime/Project-Detector-Pilot-20260907/annotation-goal77/verification-summary.json) · [回执](/srv/sentinel-data/VisionCortex3090Ti/Runtime/Project-Detector-Pilot-20260907/goal77-work-receipt.json)。两仓原分支/SHA及未提交修改保留。

## 2026-09-10 goal76：10张原图新增321框，完成手持枪与密集枪头双遍复核

PROGRESS / PARTIAL_EVIDENCE，总目标active。实际初看10原图/35裁剪；保存后实际重开10原图/10叠框及8份密集细节，修正13框边界、2处本帧已无玻璃的错误未知区，并实看6份最终叠框/1管体细节。T079架上9枪+手持1枪、23支蓝色枪头按8/8/7逐件，空孔不补件；模糊黄色枪头和透明管盖状态保留未知。新标321框/33未知区。项目417图7696框：92完整、237区域外复核、21待复核、16草稿、51排除；审计1487条。浏览器T079实际显示版本2、64框/5未知及最新统计。

417来源/分区、407条其他标注、116条旧训练版本不变；同记录当前审计检查0阻断，普通全项目完整导出仍634阻断。新图继续未映射/unassigned，无新训练/导出/模型调用，累计66次，FP v24/TP v43，生产资产不变。整体优于生产与完整原片多机位/语义/归档验收仍NOT_PROVEN。继续16张草稿、21张待复核和来源核定，再冻结新训练批次。

[报告](/srv/sentinel-data/VisionCortex3090Ti/Runtime/Project-Detector-Pilot-20260907/annotation-goal76/report.md) · [核验](/srv/sentinel-data/VisionCortex3090Ti/Runtime/Project-Detector-Pilot-20260907/annotation-goal76/verification-summary.json) · [回执](/srv/sentinel-data/VisionCortex3090Ti/Runtime/Project-Detector-Pilot-20260907/goal76-work-receipt.json)。两仓原分支/SHA及未提交修改保留。

## 2026-09-09 goal75：12张原图新增302框，修正6处移液枪指托边界

PROGRESS / PARTIAL_EVIDENCE，总目标active。12原图/39裁剪实际初看，初标保存后12原图/12叠框实际重开，4份枪架细节确认两图各前三支指托初框裁切，修正6框并实际查看最终叠框。保留54未知区，不凭空孔、邻帧或遮挡猜器材。项目417图7375框：92完整、227区域外复核、21待复核、26草稿、51排除；审计1467条。浏览器T070实际显示37框/6未知、版本2及最新统计。

417来源/分区、405条其他标注、116条既有训练记录核验不变；同记录当前审计检查0阻断，普通全项目导出仍634阻断。新图继续未映射/unassigned，无新导出、训练或模型调用，累计66次，FP v24/TP v43，生产资产不变。真实整体优于生产与完整原片多机位/语义/归档验收仍NOT_PROVEN。继续26张草稿、21张待复核及来源核定。

[报告](/srv/sentinel-data/VisionCortex3090Ti/Runtime/Project-Detector-Pilot-20260907/annotation-goal75/report.md) · [核验](/srv/sentinel-data/VisionCortex3090Ti/Runtime/Project-Detector-Pilot-20260907/annotation-goal75/verification-summary.json) · [回执](/srv/sentinel-data/VisionCortex3090Ti/Runtime/Project-Detector-Pilot-20260907/goal75-work-receipt.json)。两仓原分支/SHA及未提交修改保留。

## 2026-09-09 goal74：12张原图新增282框，密集蓝枪头逐件复核

PROGRESS / PARTIAL_EVIDENCE，目标继续active。12张原图、35份裁剪完成初看；初标保存后实际重开12原图/12叠框，再看6份细节，新增282框、保留36未知区。T056/T057每图23支蓝枪头与空孔分开，10枪、10管、3橙盖分别核对；空手套不标为已戴手，重遮挡器材不套邻帧。项目417图7073框：92完整、215区域外复核、21待复核、38草稿、51排除，审计1443条。实际工作台T057版本2、61框/5未知及新统计可见。

全部417来源/分区、其他405条标注及冻结116记录不变；同记录当前审计检查0阻断，全项目普通完整导出仍634阻断。新图未映射来源且unassigned保留，无新导出/训练/模型调用，累计66次，FP v24/TP v43及生产资产不变。独立模型质量和完整原片多机位/语义/归档验收仍NOT_PROVEN，继续剩余标注、来源核对和系统对照。

[本轮报告](/srv/sentinel-data/VisionCortex3090Ti/Runtime/Project-Detector-Pilot-20260907/annotation-goal74/report.md) · [核验](/srv/sentinel-data/VisionCortex3090Ti/Runtime/Project-Detector-Pilot-20260907/annotation-goal74/verification-summary.json) · [回执](/srv/sentinel-data/VisionCortex3090Ti/Runtime/Project-Detector-Pilot-20260907/goal74-work-receipt.json)。两仓原分支/SHA和未提交工作保留。

## 2026-09-09 goal73：7张原图新增187框，修订T002漏包侧门

PROGRESS / PARTIAL_EVIDENCE，目标继续active。8张实际原图初看、局部放大、初标保存后重开复核，共217框/26未知；T040/T041每图10支枪、10支试管及3管盖逐件确认，空孔不补数，T004收紧手/纸包遮挡边界。项目417图6791框：92完整、203区域外复核、21待复核、50草稿、51排除，审计1419条。浏览器实际显示T041版本2、37框/4未知及最新统计。

全部417来源/分区、其余409标注及冻结116记录不变；同记录当前审计检查0阻断，普通全项目导出仍634阻断。新图来源未映射、unassigned保留。本轮无新导出/训练/模型调用，累计66次，FP v24/TP v43不变，生产不替换。T002标签已修订，历史原始域分数仅适用旧标签，本轮未重评分。独立质量及完整原片多机位语义/归档验收仍NOT_PROVEN；继续剩余标注与来源核对。

[本轮报告](/srv/sentinel-data/VisionCortex3090Ti/Runtime/Project-Detector-Pilot-20260907/annotation-goal73/report.md) · [核验](/srv/sentinel-data/VisionCortex3090Ti/Runtime/Project-Detector-Pilot-20260907/annotation-goal73/verification-summary.json) · [回执](/srv/sentinel-data/VisionCortex3090Ti/Runtime/Project-Detector-Pilot-20260907/goal73-work-receipt.json)。两仓原分支/SHA和未提交工作保留。

## 2026-09-09 goal72：11张原图双遍复核，修正4张旧标注

PROGRESS / PARTIAL_EVIDENCE，总目标继续active。本轮实际查看15张完整原图、36份放大裁剪；初标保存后逐张重开原图与覆盖图复核。新增11张299框，另修正T085/T091/T094/T095；合计核对410框，保留48处未知区域。项目417图6604框，92完整复核、196区域外复核、21待复核、57待初标、51排除；审计1403条。

逐支核对每图10支移液枪及独立架；统一旧图橙盖容器类别，补T094/T095打开玻璃门的天平边界，收紧手掌/实验服/玻璃器材边界与遮挡标记。T096无法辨认的被遮挡药匙保留未知，不从邻帧补件。15图仍在未映射原始来源组且unassigned，不混入训练/验证。

全部417来源/分区与402张其他标注不变；旧116条训练/验证版本和哈希不变。旧批次因项目审计版本更新拒绝，逐条核对后仅生成当前审计检查清单，0阻断；全项目普通导出仍634阻断。本轮无新导出/训练/模型调用，累计66次训练，最新FP v24/TP v43，生产权重不替换。既有1459项回执文件实际哈希核验通过。

工作台新页面实际显示6604框、196张区域外复核，T094版本4、28框/3未知、第三人称/未分组。真实模型整体优于生产、独立质量、完整原视频多机位/语义归档验收仍NOT_PROVEN。继续57张原始草稿及来源映射、真实失效场景覆盖；本轮只改项目标注与进度文档，保留两仓分支/SHA和未提交修改。

[本轮报告](/srv/sentinel-data/VisionCortex3090Ti/Runtime/Project-Detector-Pilot-20260907/annotation-goal72/report.md) · [核验摘要](/srv/sentinel-data/VisionCortex3090Ti/Runtime/Project-Detector-Pilot-20260907/annotation-goal72/verification-summary.json) · [回执](/srv/sentinel-data/VisionCortex3090Ti/Runtime/Project-Detector-Pilot-20260907/goal72-work-receipt.json)。

## 2026-09-09 goal71：v43状态补充训练完成，局部有进有退

PROGRESS / PARTIAL_EVIDENCE，整体goal继续active，生产不替换。本轮未改标注，仍417图6305框（92完整/185区域外/21待复核/68草稿/51排除）。116条固定批次=114训练父图+2原验证裁剪，456增强+114原图=570训练文件；旧107条及1054份旧图像标签不变，9份36增强对照实际查看。

TP v43从生产初始化，960 FP32 batch2/freeze10/AdamW LR.0001/image_uniform/one2many-only/20轮，实际5700更新、11400访问、302.91秒，退出0。旧父图访问不变，各100；新增9图各100，故总更新及优化顺序不同。累计66次完整训练（FP23/TP43），最新FP v24/TP v43。固定last.pt SHA`4d468705dc1501be4ef723703eadd1c701255adce53ad6cda8ae3e607ce6dc35`。

三权重实际390源图/390NMS，旧121图生产/v42像素与原始预测完全复现。2固定开发裁剪27实例：生产15/7/12，v42为25/2/2，v43为24/1/3；v43P96%、R88.89%、mAP50–95 76.99%（v42 78.67%），新增漏gun07。全图已知87→88/106、原始域72→73/88、新增训练89→90/90；旧训练511→506/529，枪头53→48/61。全图枪头仍0/8、原始域天平仍0/3。7份实看对照确认65秒训练纸包补回，密集/旧域问题保留。共有类别26实例对照也非全面进步；这些均不是独立质量验收。

v43不晋升，本轮无新完整视频/多机位/付费语义/正式归档运行，无产品源码修改。继续未完成精细标注、来源映射、清晰手持与多背景密集小目标，然后冻结完整原片系统对照；避免围绕重复小验证集扫参数。

[参数、图像和结果报告](/srv/sentinel-data/VisionCortex3090Ti/Runtime/Project-Detector-Pilot-20260907/annotation-goal71/report.md) · [训练计划](/srv/sentinel-data/VisionCortex3090Ti/Runtime/Project-Detector-Pilot-20260907/goal71-training-plan.json) · [本轮回执](/srv/sentinel-data/VisionCortex3090Ti/Runtime/Project-Detector-Pilot-20260907/goal71-work-receipt.json)。两仓原分支/SHA和未提交工作保留。

## 2026-09-09 goal70：完成17张原视频草稿复核，筛出训练状态代表

PROGRESS / PARTIAL_EVIDENCE；总目标继续active。17张Customflow原片既有草稿完成原图初看、密集裁剪逐件核对、保存后重开原图及覆盖图第二遍复核；新增278个实物框、保留40处未确认区域。5/10秒瓶盖修正侧壁遗漏，架上10支枪分别标注，7个可辨认管盖不按透明孔位补管。项目417张/6305框；92完整复核、185区域外复核、21待复核、68原始第三人称草稿、51排除，审计1373条。所有原视频Customflow既有草稿已完成区域外复核，原始第三人称数据仍未全部完成。

保留既有114行训练批次版本，只加入60秒（枪架移走、杯仍在）及85秒（枪头盒和样品瓶移走）的状态代表；其余15个相近静止帧留工作台，避免重复加权。冻结116行=114训练父图+2原固定完整验证裁剪；批次检查0阻断、部分监督导出及VisionCortex消费方实际验证116行通过。全项目普通导出仍634阻断，没有删除未知区域或挪动June17验证组的7张待复核移液图。当前新批次无增强、尚未训练。

导出`/home/x1/.local/share/annotation-workbench/exports/tp-state-representatives-goal70`，回执SHA`91858397c8e18f37e0dbcd7e832c61e2698b816ebbf4ac23b42366d4aeeaa7aa`。新开工作台页面实际显示6305框、60秒11框/2未知/版本3/第三人称/train。原T045旧页面仍持旧JS统计，不据此宣称旧页面已自动刷新。

本轮无新模型调用或训练，累计仍65次完整训练（第一人称23、第三人称42），最新FP v24/TP v42，生产权重不替换。新增静置状态未解决真实手持枪缺失，不能称准确率提高。下一轮按冻结批次设计与v42同设置的状态补充训练，并检查空孔误报、旧类退步和真实视频；尚余68张原图草稿及来源映射、独立逐类质量、完整多机位系统/正式归档门禁。

详情：[本轮标注与批次报告](/srv/sentinel-data/VisionCortex3090Ti/Runtime/Project-Detector-Pilot-20260907/annotation-goal70/report.md)；[本轮回执](/srv/sentinel-data/VisionCortex3090Ti/Runtime/Project-Detector-Pilot-20260907/goal70-work-receipt.json)。两仓当前分支/SHA和未提交改动保留，本轮仅更新标注与进度记录，无产品源码修改。

## 2026-09-09 goal69：七帧双遍复核与上下文模型对照

PROGRESS / PARTIAL_EVIDENCE；总目标active，生产权重未更换。本轮7张既有Customflow训练帧完成实际初标、保存后重开原图复核，新增72框（含10支独立移液枪），保留11个未知区域。项目417图/6027框，92完整复核、168区域外复核、21待复核、85草稿、51排除，审计1339条。剩余17个Customflow草稿和68个未映射第三人称原图草稿；普通导出仍634阻断。

既有107个训练版本哈希不变，加入7张后冻结114行（112训练父图+2固定完整验证裁剪），部分监督导出与VisionCortex消费方验证通过，无增强、未知区域完整保留。导出`/home/x1/.local/share/annotation-workbench/exports/tp-context-states-goal69`，回执SHA`b5dae77e23bf061046bad92df7f6ae09bfce38e7e2078ad89646f2b1fc46d6fe`。该批次尚未训练。

固定960/FP32/batch1/.25报告阈值/.7NMS/.5匹配IoU，三权重实际21次源图调用与21次NMS：72个已知实例生产45、v41 70、v42 71；其中枪10件分别10/9/10。均为已有训练来源局部诊断，不是全图准确率或独立质量；旧21类与新23类须区分。三模型均漏65秒遮挡纸包，v42在模糊枪头ignore内有重叠预测。工作台新页面实际显示6027框、当前20秒21框/3未知区域、移液枪10件。

未改实现源码、未新增训练、未替换生产模型/引擎；真实训练累计65（FP23/TP42），最新FP v24、TP v42。完整原视频多机位、独立逐类质量、关键素材和正式归档门禁仍未完成；继续剩余精细标注/来源映射后进行有针对性的训练。详情与已查看对照图见：

- `/srv/sentinel-data/VisionCortex3090Ti/Runtime/Project-Detector-Pilot-20260907/annotation-goal69/report.md`
- `/srv/sentinel-data/VisionCortex3090Ti/Runtime/Project-Detector-Pilot-20260907/diagnostics-goal69-context/summary.json`
- `/srv/sentinel-data/VisionCortex3090Ti/Runtime/Project-Detector-Pilot-20260907/goal69-work-receipt.json`

## 2026-09-09 goal68 最新续接

PROGRESS / PARTIAL_EVIDENCE，goal保持active。4张原始第三人称图T045/T085/T094/T095完成初标与保存后实际重看原图的区域外复核，新增117框；项目417图/5955框/1325审计，完整92、区域外161、待复核21、草稿92、排除51。来源未映射四图继续unassigned；普通导出634阻断保留。

固定121图×3权重×3输入尺寸实际1089次源图推理/NMS，960完整复现goal67；v42输入960/1280/1920的2裁块TP/FP/FN分别25/2/2、21/4/6、7/8/20。3张全图已知命中87/106、87/106、48/106；旧域72/88、82/88、82/88。8个验证枪头全配置仍0命中。不能直接上调全局分辨率或宣称超过生产/泛化达标。本轮0新训练，累计65次，FP v24、TP v42；生产权重与引擎未替换。

工作台loadImage同步刷新列表与统计，实际浏览器验证外部保存T095后点击重新读取，5927→5955且当前T094版本未变。75项pytest、node语法、两仓diff检查通过。70个本地原视频采样时刻及72张原图缩略图检查完成，没有新增可确认手持枪训练样本；未作整段连续视频或端到端质量宣称。

详细报告：`/srv/sentinel-data/VisionCortex3090Ti/Runtime/Project-Detector-Pilot-20260907/annotation-goal68/report.md`；最终回执：`/srv/sentinel-data/VisionCortex3090Ti/Runtime/Project-Detector-Pilot-20260907/goal68-work-receipt.json`。下一轮沿已保存版本继续细标与来源映射，扩大真实失效场景后再设计训练。完整多视角/独立质量/云端语义重跑/正式归档等仍待验收。

---

# VisionCortex 工作交接 — 2026-09-07

## goal67：显式水平翻转与v42真实对照（2026-09-09）

**PARTIAL_EVIDENCE；v42不晋升，goal继续active。** 新增显式增强策略v2，图像/实物/未知区同步反射，v1保持复现。仍417图5838框、审计1317；新增标注0。105训练父图+420增强（210翻转）=525文件，原107条批次与214份原始图像标签不变，210未翻转派生像素/标签不变。4组16预览实看，工作台75/训练相关104测试、ruff/compileall通过。

v42从生产21→23初始化，960 FP32 batch2、freeze10、AdamW LR.0001、image_uniform、one2many-only、20轮，实际5260更新/10500访问/276.07秒，退出0。累计65次完整训练（FP23/TP42），FP仍v24。系统分支固定2裁剪27实例：生产15/7/12、v41为23/1/4、v42为25/2/2；v42 P/R均92.59%、mAP50–95 78.67%。仅小开发集；原始域88已知77→72、天平1/3→0/3，公开新图4已知1→0，不能称整体优于生产。

363次固定图片调用+24次公开图调用均核验；生产/v41原像素和原始预测完全复现。95.03秒开发派生视频760帧/190批/10.71秒；760像素/PTS身份一致，2280阶段重算一致，复跑0调用，换权重启动前拒绝。实际15份对照显示手持白枪仍漏、空孔仍误报，v42枪头预测7081次不等于正确实例增加。完整原片多视角、真实语义与归档、独立质量/稳定发布仍NOT_PROVEN。

[本轮参数、图像、代码与训练报告](/srv/sentinel-data/VisionCortex3090Ti/Runtime/Project-Detector-Pilot-20260907/annotation-goal67/report.md) · [训练计划](/srv/sentinel-data/VisionCortex3090Ti/Runtime/Project-Detector-Pilot-20260907/goal67-training-plan.json)。保留两仓分支/SHA和生产资产；改动仅增强生成/严格消费/相应测试文档。source_frames未改，另任务最终集成修复指针仍ece199a3，未覆盖或宣称本机已合入。下一步补真实手持/空孔/密集目标及完整来源隔离验证，继续迭代。下列旧“当前”均为历史记录。

## goal66：新来源与泛化核对（2026-09-09）

**PARTIAL_EVIDENCE；goal继续active，候选不晋升。** 本轮未重训：实际核对30张本地抽帧预览，筛查8张有许可的公开实验图，1张完成双遍区域外项目标注（4框/1未知）。当前417图5838框、完整92/区域外157/待复核21/草稿96/排除51，审计1317。原416条不变；108条部分监督资格检查0阻断，全项目普通导出仍634阻断。

生产/v39/v41各8图实跑，共24次源调用/NMS。新图4已知命中分别2/0/1；v41仍把玻璃器材旋塞以0.968误识为试管。公开pipette标签抽查是玻璃管/吸球，不能直接并成移液枪训练。手持枪仍仅2张原图/4裁剪，新样本覆盖不足，不能据小验证分数替换生产。最新FP v24/TP v41及64次完整训练不变。工作台实见版本3/1920×1080/4框/1未知；完整系统语义、真实归档与独立质量仍NOT_PROVEN。

[本轮详细来源、标注、实际对照及后续步骤](/srv/sentinel-data/VisionCortex3090Ti/Runtime/Project-Detector-Pilot-20260907/annotation-goal66/report.md)。保留两仓库分支/未提交工作及生产资产，本轮无产品源码改动、提交、推送、NAS或付费调用。最终集成任务回报SHA为 **ece199a3b639d8939beee4ec647322e49caff01a**，替代历史6ca2b8e；其FFmpeg6/9源帧v2修复未合入本共享7db源码，不得覆盖。下列旧“当前”均为历史状态。

## goal65：原片补标与v41（2026-09-09）

**PARTIAL_EVIDENCE，v41不晋升，goal继续active。** 新抽14帧/复用16帧，实际复核0秒和55.0196秒原图，新增39框/7未知（10支枪、7个管盖分别标注，模糊手持器材不猜标）。两图版本1→2→3，当前416图5834框、完整92/区域外156/待复核21/草稿96/排除51，审计1313。两个工作台页面及28项测试通过；全项目普通导出仍632阻断。

v41从生产21→23类初始化，105训练父图+420增强=525文件；960/FP32/batch2、AdamW LR.0001、freeze10、image_uniform、one2many损失1、20轮，实际5260更新/10500访问/256.18秒，退出0。旧105条和1034份图像标签不变；累计64次完整训练（FP23/TP41），FP仍v24。

固定2裁图27实例：生产15/7/12，v39为23/2/4，v41为23/1/4；v41 P95.83%、R85.19%、mAP50–95 78.69%（v39 80.90%）。小裁图不足独立验收。完整画面106已知85→84；原始88已知76→77但天平3→1、枪35→37/42；旧训练枪头未匹配4→27。新增两训练图36→39/39只是拟合。

95.03秒真实开发派生片段760帧/190批/8.99秒；760原像素/PTS/packet与生产/v39一致，2280阶段结果重算吻合。复跑0模型/0NMS，换权重续跑启动前拒绝。实看13份对照，白色手持枪漏检、空手套误报、密集枪头错框仍在。生产权重/引擎和两仓库当前分支/未提交工作保留，未提交/推送。下一步清晰手持与密集小目标、旧类退步分析及完整原片验收；真实语义/归档/系统浏览器仍NOT_PROVEN。

[本轮详细报告](/srv/sentinel-data/VisionCortex3090Ti/Runtime/Project-Detector-Pilot-20260907/annotation-goal65/report.md) · [参数、数据和训练回执](/srv/sentinel-data/VisionCortex3090Ti/Runtime/Project-Detector-Pilot-20260907/goal65-training-plan.json)。本轮FFmpeg4.4使用现有源帧契约v1；另任务回传6ca2b8e的v2兼容修复尚未合入共享树，后续不得覆盖该修复。

## goal64：两张原始状态图双遍补标（2026-09-09）

**PARTIAL_EVIDENCE，goal继续active。** T050/T058分别新增12/24个实例，原图与密集局部实看后初标、再重开原图复核，版本0→1→2，各保留2处未知。区分已戴手套/裸手/空手套，逐件9管3盖、架与枪头盒分开；模糊黄色枪头未猜标。当前402图5795框、完整92/区域外154/待复核21/草稿84/排除51，审计1281；其他400条记录和全部来源不变。

两图未映射来源、unassigned及used_by_baseline保持，不计新训练或验证图片。2页浏览器版本/原图/框/未知区域与API吻合，0页面错误；28项标注测试通过；普通导出仍604阻断。冻结标签后生产/v39/v40各2图实跑，共6源调用/NMS，36已知分别匹配29/34/34，只有内部已知匹配证据。T058器材后裸手候选模型仍漏，v40一根管有框但IoU0.498514未达0.5；不能当全图精确率或独立效果。

本阶段无重训，最新TP仍v40、FP仍v24，累计63次完整训练。v40原始场景退步和真实手持枪漏检仍阻止替换。下一步原始序列标注、来源核对、多状态训练样本与完整验证；全原片多视角语义/归档及系统浏览器仍NOT_PROVEN。生产资产/当前分支/未提交修改保留。

[详细标注、浏览器和对照报告](/srv/sentinel-data/VisionCortex3090Ti/Runtime/Project-Detector-Pilot-20260907/annotation-goal64/report.md) · [v40参数与视频对照](/srv/sentinel-data/VisionCortex3090Ti/Runtime/Project-Detector-Pilot-20260907/annotation-goal63/report.md)

## goal63：来源平衡训练v40与真实对照（2026-09-09）

**PARTIAL_EVIDENCE；v40不晋升，goal继续active。** 同105条批次/515训练文件/20轮/5,160更新/10,300访问，仅改为来源平衡采样，实际约25%/组；训练261.10秒。两原始手持图及4裁剪的训练访问600→846次，不增加独立样本。累计63次完整训练（FP23/TP40），FP最新仍v24。

351次固定图片预测+760帧真实95秒开发视频已完成。2完整裁剪27实例：生产15/7/12，v39 23/2/4，v40 23/1/4（TP/FP/FN）；v40 P95.83%/R85.19%、mAP50–95 71.86%，每类样本仍不足。部分整图106已知85→86，原始88已知76→70，关键类退步。实际看完13份对照，手持枪仍漏、空孔仍误判枪头、未穿戴手套仍误判手。760帧同像素/PTS，2280份阶段重算一致，复跑0调用，换权重续跑启动前拒绝；检测单次9.08秒不等于全链路验收。

包装进程143后保留训练输出并核验完整20轮，不重训、不虚构子进程退出码。全项目402图5759框/audit1277无修改；生产权重/引擎、分支和未提交工作保留。本轮无产品代码改动、付费调用或新增浏览器验证。下一步优先多状态原始样本、背景与完整验证图标注，完整多视角语义/归档仍NOT_PROVEN。

[详细参数、证据和实际对照图](/srv/sentinel-data/VisionCortex3090Ti/Runtime/Project-Detector-Pilot-20260907/annotation-goal63/report.md)；阶段回执 `/srv/sentinel-data/VisionCortex3090Ti/Runtime/Project-Detector-Pilot-20260907/goal63-work-receipt.json`。

## goal62最新进展：原片补标与v39（2026-09-09）

**PARTIAL_EVIDENCE，v39不晋升，goal继续active。** 新导入16张原片抽帧，4张完成两遍项目标注51实例，保留5个ignore；其余12张仍draft。全项目402图5759框：完整复核92、区域外复核152、待复核21、draft86、排除51（含16张既有裁剪图）。同日同实验curated-2026-06-18整体train，旧386条图片记录不变。105条批次检查0阻断；全项目普通导出仍604项阻断。

第三人称最新v39-last，累计62次完整训练（FP23/TP39），第一人称仍v24。v39从原生产21→23迁移，103训练父图+412离线增强=515训练文件，960/FP32/batch2、AdamW LR.0001、freeze10、20轮、one2many损失1/one2one0、image_uniform；实际5160次更新/10300样本访问/254.99秒。验证仍原2张裁剪27实例，新增训练前已冻结last终点，旧101条及994份图像标签不变。

系统同条件3权重各117图，共351次真实源调用/NMS。完整2裁剪：原生产15/7/12（TP/FP/FN），v38为24/3/3，v39为23/2/4；v39 P92.00%/R85.19%、mAP50–95 80.90%。部分整图106已知：60→87→85；T000–T002的88已知70→72→76；goal60六原图157已知125→148→148；新训练51已知38→44→51。训练拟合与小裁剪指标不能代替泛化，手/瓶盖等有退步。

95.0279秒真实开发片段8fps共760帧/190批/9.26秒，源像素及PTS/packet与生产/v38一致，2280份阶段结果重算吻合；复跑0模型/0NMS，换权重续跑启动前拒绝。看完7时刻+2细节+7静态对照：手持白色移液枪仍漏、密集枪头仍错框，T002未穿戴手套仍误检。当前共享代码增加了计时，已保留差异，不拿不同版本耗时声称速度提升。

工作台4张新标注浏览器0错误且版本/原图/中文实例/ignore与API吻合；完整系统真实多视角播放、语义和归档仍NOT_PROVEN。保留生产权重/引擎/当前分支与未提交修改，本轮无产品源码改动或付费请求。下一步针对手持遮挡与小目标状态/几何变化、采样增强的泛化；继续来源映射、原始标注与完整原片的分阶段验收。

完整参数、对照与图像：[goal62报告](/srv/sentinel-data/VisionCortex3090Ti/Runtime/Project-Detector-Pilot-20260907/annotation-goal62/report.md)；回执根`/srv/sentinel-data/VisionCortex3090Ti/Runtime/Project-Detector-Pilot-20260907`中的`goal62-work-receipt.json`，训练计划`goal62-training-plan.json`，数据检查`annotation-goal62/`，运行`third-person-customflow-context-v39/`、`diagnostics-goal62/`、`candidate-video-goal62/`、`video-review-goal62/`。主权重SHA`114a13b16209e6141ef0874abd285e4e13c9a412b657aeb91eec2295c180c70f`。后续历史记录中的“当前”只适用于当时版本。


## 1. 接手结论与用户目标

这是一份本机继续开发/实测的交接，不是生产验收通过证明。交接时只核对了本地 Git、服务状态、GPU 进程和已有回执；没有启动模型、调用豆包、扫描 NAS、重跑测试或发布版本。

用户最终要求：充分发挥本机 RTX 3090 Ti 的有效硬件性能，模拟普通用户真实端到端流程，从网页上传视频或选择 NAS 采集批次开始，自动完成分析、证据生成、关键素材、报告、可溯源归档及网页查阅。无需人工审批；个别素材失败要自动隔离，不能使整批无条件崩溃，也不能把失败或不确定素材当成已证实动作。

必须清楚报告：输入几路、每路多长、合计视频时长、共同实验时间轴、冷/热缓存、预处理时间、完整流水线时间、用户提交至正式归档可查阅时间、GPU/CPU/解码/编码/磁盘指标、真实模型调用及服务端 Token。不能把预处理 20 分钟说成全链路 20 分钟，也不能用短视频线性换算声称六路长视频已达标。

**最终目标仍为 `NOT_PROVEN`。最近实际豆包请求返回欠费错误；当前合并版本还没有完成真实网页提交至正式归档的全链路验收。**

## 2. Git 状态：保住本机未推送工作

- 工作目录：`/home/x1/Projects/VisionCortex`。
- 当前分支：`codex/rtx3050-device-delivery-20260904`。
- 当前 HEAD：`7db0dcd052891e0e189c5e931a27558256fea032`。
- 合并前本机父提交：`7c5f3030a95a74912dd11c2eb405fc9c66b56415`。
- 已合入 4060 分支：`codex/input-alignment-identity`，SHA `b2a3c7df47a3d66e2554fbb6be33c9826584ed95`。
- 最新这一轮远端增量实际是两个提交：`6a7f823`（发布/证据包加固）和 `b2a3c7d`（归档检索/交付加固），不是十个新增提交。“十阶段”是用户对累计改动的描述。
- 开发远端：`development = https://github.com/kealan-Jun/VisionCortex.git`。
- 稳定远端：`origin = https://github.com/RealityLoopAI/VisionCortex.git`。**不要因为它叫 origin 就向它提交开发改动。**
- 本地跟踪的 upstream：`development/codex/rtx3050-device-delivery-20260904`，SHA `4a33a253b338dda5e71ec9f840c9d799680029bf`；本地领先 30 个提交，尚未推送本次合并。
- 本地缓存的 `development/main`：`f8801cbfc2486fb73f78d66948b1adb50129c975`。9 月 5 日核对的远端默认分支是 main；9 月 7 日未 fetch/ls-remote，不能把缓存当作最新远端状态。本地没有 `development/HEAD` 符号引用。

用户未提交工作必须保留：

- `README.md` 原有新增 14 行。
- `docs/VisionCortex-交付验收单模板.md`。
- `docs/VisionCortex-历史真实六路验收示例.md`。
- `docs/VisionCortex-普通用户10分钟操作指南.md`。
- `docs/VisionCortex-端到端视频分析交付手册.md`。
- `docs/assets/`。

临时 README 保护 stash 仍在：`c746e34648e2d04d404677f20cf2991d22964b1f`，9 月 7 日为 `stash@{0}`。已成功 apply，恢复后的 patch-id 与 stash 一致；**不要再次 apply**。本次交接不清理它。不要 reset/checkout 覆盖用户文件，也不要 `git add .` 或 `git add -A`。

## 3. 已完成的实现与验证边界

### 合并后的功能

- 上游：证据包 fail-closed 发布、代码/配置/模型/产物哈希溯源、跨文件契约、原子正式版本指针及恢复；自动验收和日报 V2；本地可重建 SQLite 检索目录、分页、中文检索、release_id 绑定媒体 URL、传输并发限制；可选多人角色和访问审计。
- 保留本机：3090 Ti 性能参数、解码器复用、关键素材编码重叠、Ark 失败熔断和事件隔离、未完成任务的初步产物展示、NAS 相机发现和时间匹配、桌面自启与大屏/窄屏适配。
- 合并补齐：分页摘要默认值、真实总数、报告加载、档案状态更新、release 缓存失效、素材/报告库独立分页状态、缺失焦点事件不误跳到其他素材、视频不预加载正文。
- 合并相对于本机前一 HEAD，没有改动 `configs/rtx3090ti-ubuntu-production.yaml`、`src/visioncortex/archive.py`、`src/visioncortex/mllm.py` 的已有本机优化。

### 9 月 5 日验证记录（不是 9 月 7 日重跑）

- `PROVEN`：本地确定性检查 775 项收集，771 passed、4 skipped；`node --check src/visioncortex/web/app.js`、`ruff check src tests`、`.venv/bin/python -m compileall -q src tests`、`git diff --check` 均通过。
- 相关新增回归：`tests/test_web_archive_pagination.py`、`tests/test_evidence_indexing.py`。
- `PARTIAL_EVIDENCE`：浏览器实际验证了档案、六个素材、筛选、报告入口及 3840×2160 / 390×844 视口无横向溢出，没有捕获到 JavaScript 错误。使用的是 `VC-LOCAL-SIX-VIEW-ACCEPTANCE` **合成档案**，只证明界面/结构，不是真实实验质量；不声称所有浏览器全部适配完成。
- `NOT_PROVEN`：合并 SHA 的真实模型全链路、真实动作准确率、真实 NAS 正式发布、六路长视频性能、跨平台 CI 与稳定发布就绪。

旧合成档案动作键仍可出现 `liquid_transfer`，与新本体 `liquid_movement` 有历史别名差异。不要把选中旧键成功说成新键完全兼容；旧档案也不是生产准确性基准。

## 4. 本机运行与凭据边界

- GPU：RTX 3090 Ti，`nvidia-smi` 报告显存总量 24564 MiB。9 月 7 日交接快照没有 CUDA 计算进程；桌面仍使用部分 GPU。这不代表整个产品目标完成，也不授权启动其他训练。
- 生产解释器：`/srv/sentinel-data/VisionCortex3090Ti/.venv/bin/python`。
- 9 月 5 日实测依赖：Python 3.12.13、Torch 2.6.0、Ultralytics 8.4.28、TensorRT cu12 10.16.1.11、httpx 0.28.1。TensorRT 发行包名为 `tensorrt-cu12` / `tensorrt-cu12-bindings`，不能因查不到名为 `tensorrt` 的 distribution metadata 就误判未安装。
- 生产配置：`configs/rtx3090ti-ubuntu-production.yaml`；本地无 NAS 配置：`configs/rtx3090ti-ubuntu-local.yaml`。
- 模型/引擎：`/srv/sentinel-data/VisionCortex3090Ti/Models/ClosedSetYOLO/` 与 `/srv/sentinel-data/VisionCortex3090Ti/Engines/`。真实运行前重新核对身份/哈希/运行环境，不能只看文件存在。
- `visioncortex-local.service`：9 月 7 日 active；既有入口 `http://127.0.0.1:8000/#/home`，独立无 NAS 合成验收/备用服务，MLLM 关闭。
- `visioncortex-analysis.service`：9 月 7 日 active；既有入口 `http://127.0.0.1:8001/#/home`，NAS 生产配置及监控服务，MLLM 开启。9 月 5 日两项服务均已重启加载合并版本，9 月 7 日没有重启。
- 9 月 5 日生产队列无 queued/running；9 月 7 日没有重新读取队列。接手先检查队列及 GPU 所有者，不要在工作中重启/抢占。
- NAS 正式归档配置根：`/home/x1/桌面/nas/VisionCortexExperimentArchive`；正式 staging 是其中 `.VisionCortex-Run-Staging`；持久缓存：`/home/x1/桌面/nas/VisionCortexExperimentCache`。
- 本地 Runtime：`/srv/sentinel-data/VisionCortex3090Ti/Runtime`，NoNasWeb 必须保持所有存储根不继承 NAS。

Ark 凭据只经 `src/visioncortex/credentials.py::ensure_ark_api_key` 从安全环境/批准的本机文件加载。文件路径 `/home/x1/.config/VisionCortex/ark_api_key`，9 月 5 日验证 owner、普通文件、0600 和格式；不要打印文件、进程环境或请求 Authorization，不要把密钥写入回执/文档/测试。

**最近一次实际 Ark 检查：2026-09-05，生产解释器和生产模型配置，纯文本连通性请求，无实验图像；HTTP 403，错误码 `AccountOverdueError`，耗时 0.169 s，usage 为 null。9 月 7 日未重试，不能宣称账户现在仍欠费或已经恢复。**

接手先做一次安全、有界的账户连通性预检。若仍欠费，明确要求用户恢复方舟账户，不要购买/充值，不要反复启动注定不能完成语义证据的长任务。可独立推进不依赖 Ark 的本地瓶颈检查，但必须标为局部性能证据。

## 5. 输入、拷贝与历史真实视频回执

本地运行证据根：`/srv/sentinel-data/VisionCortex3090Ti/Runtime/RealVideoValidation-20260904`。以下均为已有回执；交接只读取本地文件，没有复查远端媒体。

### U 盘数据已留存的证据

- 原入口 `/media/x1/USB/video`；本地副本 `/srv/sentinel-data/VisionCortex3090Ti/Runtime/USB-Imports/2026-09-04/video`。
- `nas-index/usb-nas-copy-receipt.json` 记录：149 个文件、11342242051 字节，源/目标数量及大小相同；递归 checksum dry-run 差异为空，`status=verified`、`usb_safe_to_remove=true`。这是 9 月 4 日复制完成的历史证据，不是本次又复制了一遍。
- NAS 已命名位置记录为 `/mnt/realityloop-nas/VisionCortexValidationSources/2026-09-04_USB_Import/`，三组：`01-Raw-Four-Camera-RealityLoop-20260616`、`02-Raw-CustomFlow-ACEBDF-0002`、`90-Reference-Existing-Derived-Experiments-20260803`。
- 第 90 组是既有衍生产物参考，不能当作原视频或独立真值。
- 旧路径与生产配置路径可能来自既有不同挂载/解析入口。接手用现有配置和 resolver 检查，不要自行改写冻结路径或为省事复制原片。

### NAS 索引批量验证

- 本地汇总：`nas-index/nas-validation-consolidated.json`。
- 历史源索引：`/mnt/realityloop-nas/experiment_videos_1h/index.csv`；派生索引在本地 `nas-index/experiment_videos_1h-derived-index.csv`。
- 回执的 45 个实验记录：30 个清单预检可运行，15 个缺少准确配对来源；运行后 8 个流程完成，22 个输入/时间对齐门禁失败，合计 37 个未完成。
- 这些运行使用 `2cc70b8` / `cb01d47`，不是当前合并 SHA；批量验证关闭了 Ark，不能称为 45 个真实端到端通过。
- 回执 `timeline_seconds=4676.565` 是该索引汇总口径，不是“六路每路 1 小时”。需查看各清单实际时长后再报告。
- 典型失败：CSV 不足两行、第一/第三人称没有相同时段覆盖。不能拿其他时间相机视频凑齐视角绕过门禁。

### NAS 实时相机监控

生产配置按顶层 `*_cam*` 自动发现目录，30 s 轮询、120 s 稳定等待、每相机近期 32 个 recording，有界元数据扫描。按 recorder 的 session 身份及跨相机时钟重叠配批次，不是目录中任取最近视频。

9 月 5 日健康信息显示 9 个相机目录、monitor=watching、连续失败 0；其中 `lubancat-52d2ef0c_cam01`、`orangepi5pro-fe0f7222_cam01` 缺角色配置，必须显式阻塞，不猜第一/第三人称。9 月 7 日没有重扫相机目录；不要把 9 当永久固定数量。仍需模拟符合采集协议的写入，验证稳定后自动出现批次与正确时间配对。

## 6. 性能历史口径：不要误报

可核对的四路真实输入清单：`input-manifests/usb-customflow-acebdf-0002.yaml`，实验 `usb_customflow_acebdf_0002`，1 路第一人称 + 3 路第三人称，4.728 GB 原视频。

`runs-opt-reader-1bd7f0e-cold/usb_customflow_acebdf_0002/JSON-Config-Files/input_volume_report.json` 记录各路 899.234、899.965、899.773、899.953 s；合计 3598.925 s，约 59 分 59 秒视频量，但共同实验时间轴仅约 15 分钟。

同目录 `run_metrics.json`：

- 流水线内部总时长 **553.576401 s（约 9 分 14 秒）**。
- 已记录的预处理口径 **184.500462 s**。
- 精扫 163.041479 s，实验裁片 86.079566 s，关键素材 247.815280 s。
- MLLM 阶段虽标 completed，但调用数为 0、Token 为 null，**不证明实际完成豆包推理**。
- 历史对话曾报约 557.15 s；交接核对到的权威内部指标是 553.576401 s，不能混用未核对的外层耗时，更不能把它说成用户上传至正式归档全耗时。
- 这是 `1bd7f0e` 历史局部证据，不是当前 `7db0dcd` 真实全链路成绩；目录名 cold 也不能替代对缓存回执的检查。

用户记忆中的“六路每路 3.3 小时，20 分钟”在本任务尚未复现。必须找回准确原始清单和对应回执，核对 3.3 h 是每路还是合计、20 min 是预处理还是全链路；找不到就明确缺失，不能拿重复拼接短片替代。

保留现有实测调参依据：

- 引擎动态 profile 最大 batch 4，运行请求 batch 16 会被拆分；此前动态最大 batch 16 的构建尝试 OOM。盲目增大 batch 不等于有效提高吞吐，不要直接重建替换生产引擎。
- GPU resize 曾改变检测输入分布并使边界偏移超标，因此生产仍 CPU scale；不能为利用率数字降低质量。
- 实验组 aligned 编码重叠 A/B 无收益，当前 `overlap_aligned_experiment_clips=false`；关键素材 aligned 重叠保留。
- 下一步以真实阶段遥测确定 CPU resize、解码、细扫、帧读取、NVENC/NAS I/O、模型排队等瓶颈；固定输入和质量约束做有界 A/B，记录资源利用率和实际耗时，不能用单张 nvidia-smi 截图证明“已打满”。

## 7. 新对话推进顺序

1. 先读仓库 `AGENTS.md`、`README.md`、`docs/DUAL-REPOSITORY-RELEASE-POLICY.md` 及本交接。检查本机分支、工作树、队列和 GPU 所有者；不要回到 main 丢掉未推送合并，不要无故新建从旧默认分支开始的工作树。
2. 安全验证 Ark 可用性；核对 SHA、真实解释器、配置、模型/引擎哈希、输入清单、存储根、容量和独立输出目的地。真实运行与回执绑定这些身份。
3. 用现有原片走真实网页上传入口及 NAS 批次入口，验证续传、输入封条、排队、进度、自动处理、单事件隔离、报告、原子归档、哈希复核及网页检索/播放/下载；不要只跑 CLI 然后称为“模拟用户全流程”。浏览器工作要读可用 browser 技能。
4. 输入配对、时间对齐、自动证据门禁不可绕过。单事件可隔离，整批缺合法跨视角时间覆盖则必须报告输入失败。自动可信产出与模型准确率是否有独立真值测量是两个结论；不要恢复人工审批兜底，也不要删除溯源/质量门禁。
5. 在可信输入和输出约束下做 3090 Ti 性能 A/B；冷运行/复用运行分别计量，不挪用缓存成绩。先四路已知输入，再真实更长/更多路输入；不能外推六路长视频达标。
6. 补齐 NAS 不同时间窗口不混批、所有已注册相机发现、写入稳定成批、服务/任务恢复等场景。发现源文件或角色缺失，只修复有依据的配置，需用户确认的采集身份不要猜。
7. 结束时逐项给出 `PROVEN / PARTIAL_EVIDENCE / NOT_PROVEN`，附真实路径、代码 SHA、模型调用/Token、路数/时长/耗时、失败/隔离数量及缺失门禁。稳定发布按双仓策略单独执行，当前没有晋升授权或通过证据。

历史 GPU 协调：另一训练任务 `01a064e9-fe5b-7d73-bd02-9b277c35ee06` 曾明确暂停 GPU 训练、只做 CPU 数据治理，要求本任务全部完成并释放 GPU 后告知。当前只是交接且目标未完成，不要发“全部完成可启动训练”；真正准备使用/释放 GPU 时先重新核对协调状态。

## 8. 本次交接交付范围

仅新增本文件，保留所有既有修改，没有 Git 提交/推送，没有新建任务，没有运行或终止训练。用户可以在本项目新对话中发送：

> 请读取 `/home/x1/Projects/VisionCortex/docs/WORK-HANDOFF-20260907.md` 并接手当前工作。保留本机分支与未提交修改，先检查 Ark 和运行环境，再继续 3090 Ti 性能优化及真实网页/NAS 端到端自动可信证据与溯源归档验收。不要把历史测试、短视频或合成档案当作当前全链路达标证据。


## 9. 持续目标续接入口（2026-09-08，goal25）

以上为原始交接记录，后续状态以[工作进度](WORK-PROGRESS-20260907.md)、[模型迭代记录](MODEL-ITERATION-RESULTS-20260907.md)及运行根`Project-Detector-Pilot-20260907/goal25-work-receipt.json`为准。用户已授权ChatGPT逐件初标、第二遍复核、分人称训练、增强和持续迭代，并已设置active goal；当前尚未达标，不能重复新建goal或声称全部完成。方舟未恢复期间继续独立工作，不反复发起欠费预检或覆盖已完成产出。

最新已完整训练47轮（FP23/TP24），旧中断另列。v24的1280训练仅改善部分小枪头，完整整图/连续视频门禁未过，生产权重及当前分支/未提交修改保持。数据212图3079框，86完整、45仅区域外、24待复核、6保留测试草稿、51排除；F035已续接到版本5但仍holdout。优先读取最新冻结回执和状态，继续未完成标注、训练来源覆盖及真实系统验证；不得将旧候选浏览器播放或历史性能当作新模型验收。


### goal26 接续更新

用户将后续提供第三人称原始数据集，当前不得把整理视频抽帧称为已收到该原始包。F082/F083实际双遍区域外复核至版本6，保留原val；当前212图/3079框、86完整、47区域外、22待复核、6测试草稿、51排除。最新源快照在工作台`review-goal26/snapshot-after.json`，统计`counts-goal26.json`，本阶段无新训练，47完整训练保持。公开来源只完成元数据初筛，待实图、来源及全类别检查后才能采用。接续以运行根`goal26-work-receipt.json`及`goal26-public-source-research.json`为本阶段证据，保留全部未知和保留测试分组；继续独立工作，Goal active/PROGRESS，生产不晋升。


### goal27 接续更新：第三人称原始包已经收到

用户本次提供/home/x1/桌面/third_images.rar，实际169原图、无标签，已解包校验并导入现有工作台T000–T168。原始源根在RealVideoValidation-20260907/Motion-Delivery-20260907/Original-Datasets/Third-Images，包含导入/图片清单/摘要与实际初看回执。不要再让用户提供同一包，也不要将导入等同标注完成。T024双遍32已知/3未知、版本2、third_person/train，仅区域外完成；另168新图仍draft/unknown/unassigned。优先续接其他清晰的新TP来源与密集实例，核对同源跨视角分组，不按帧随机切分。

最新381图/3106框，87完整、51区域外、18待复核、174草稿、51排除，370源图/11裁剪。旧F086/F087/F117/F118已续至5/4/5/4，F117完整，其余未知保留。普通导出643阻断。读取工作台review-goal27/snapshot-final.json、counts-goal27.json及运行根goal27-work-receipt.json；47完整训练不变，本阶段无新训练/模型调用或生产替换。当前架构建议继续分人称检测、关联后互补证据；统一模型待固定对照，不得称已有最优证明。Goal active/PROGRESS，保留分支HEAD和全部并行未提交修改，继续局部工作及后续完整质量门禁。


### goal28 接续更新

T006/T025/T026/T037已完成初标与第二遍区域外复核，版本均2、15/21/22/21件；不要重复覆盖。所有原来源保持，T037来源仍未映射/unassigned，不混入训练。新包已5图区域外/111件，另164草稿；全项目381图3185框，87完整、55区域外、18待复核、170草稿、51排除，普通导出640阻断。读取工作台review-goal28/snapshot-final.json和visual-review-receipt.json、counts-goal28.json、运行根goal28-work-receipt.json；额外核验脚本第一次漏加RGB尺寸前缀已修正，仅核验错误，原始文件未变。

当前无新导出/训练/模型调用，完整47轮、生产4资产保持。保留分支HEAD及并行修改，继续剩余新原图规范标注和源组核对，再冻结新批次受控训练。分人称模型是当前基线，关联后互补证据；统一模型没有对照结果，不宣称哪种必然更准。Goal active/PROGRESS，真实质量仍PARTIAL_EVIDENCE、完整系统验收NOT_PROVEN。


### goal29 接续更新

T008/T015/T027/T031已双遍区域外复核至2，17/23/25/43件。最新381图3293框，87完整、59区域外、18待复核、166草稿、51排除，普通导出636阻断。第三人称新包9图区域外/219件、160草稿。T122/T150复杂操作组实际初看并与既有FP同源组核对，两图仍未标；续接暗区密集枪架逐件与第二遍，再继续来源覆盖。T031可辨13枪头口沿、2持枪，T027用F107实际近时刻原图澄清管口但未证明自动同步。

读取运行根goal29-work-receipt.json、goal29-complex-source-review.json，工作台review-goal29/snapshot-final.json与visual-review-receipt.json、counts-goal29.json。所有原来源和其他377条保持；无新模型调用/训练/导出，累计47完整训练不变，生产资产未替换。保留分支HEAD及并行未提交修改。Goal active/PROGRESS，质量仍PARTIAL_EVIDENCE、最终系统验收NOT_PROVEN。


## goal30 接续更新

T122/T150已双遍区域外复核至版本2、24/21件、各5处未知，共新增45。T122可辨8枪；T150撤回右缘3个不可靠初标对应，留下4已辨枪及明确未知区。41次实看；同组T123/T151/F011仅上下文未改，其他379条及全部来源保持。当前381图3338框：87完整、61区域外、18待复核、164草稿、51排除；新TP包11图区域外/264件、158草稿。普通导出634阻断，浏览器T122实际显示24件/5未知、8枪及3338框。

TP train候选26记录含21原图/5裁剪，4源组，较上次训练新10原图；待冻结新批次后受控训练，统一/分开模型公平比较仍待完成。无新增训练/模型调用/导出，累计47轮和生产4资产保持。分支codex/rtx3050-device-delivery-20260904、HEAD 7db0dcd052891e0e189c5e931a27558256fea032及并行未提交修改保留。读取goal30-work-receipt.json、goal30-next-cohort-assessment.json与review-goal30/snapshot-final.json。Goal active/PROGRESS；质量PARTIAL_EVIDENCE，最终系统验收NOT_PROVEN。


## goal31 接续更新

已冻结并检查原16train+2val与新增10原图的26train+同2val，实际查看40新增强及4密集分区，导出/消费方哈希与未知区域契约通过。v25/v26各完成520参数更新、1040样本访问，49完整训练（FP23/TP26）。实际系统分支31图×4权重=124源预测，独立重算一致：新训练图99/243→230/243，但固定full-val已知82/106→76/106；T03113枪头全漏，1280/1920额外4图调用仍无改善。生产权重保持，NOT_PROMOTED。

下一步T031密集枪头区域制作有溯源训练裁剪并实际双遍复核，保留父图未知；调整稀少目标支持及旧来源训练覆盖，固定验证继续对照。工作台381图3338框、87完整/61区域外/18待复核/164草稿/51排除不变，全项目普通导出634阻断。分支codex/rtx3050-device-delivery-20260904、HEAD 7db0dcd052891e0e189c5e931a27558256fea032与并行修改保留。读取goal31-work-receipt.json、goal31-comparison.json、goal31-training-verification.json、iteration-summary-goal31.json。Goal active/PROGRESS，质量PARTIAL_EVIDENCE，全部数据及完整系统验收NOT_PROVEN。


## goal32 接续更新

新增T031有溯源训练裁剪awcrop-9fe263d6c298e70c6f92c8ee已实际双遍复核至版本3、13枪头+1盒；纠正外框相交但无枪体的裁剪草稿。原381条及全部来源保持。当前382图3352框、370来源/12裁剪，88完整/61区域外/18待复核/164草稿/51排除；新TP原169图仍11区域外/264框、158草稿。普通全项目634阻断，限定29条v2批次检查/导出/实际增强预览通过；8010实际显示版本3及当前统计。

TP v27均匀采样与v28加裁剪完成，累计51轮（FP23/TP28）。128源图系统预测/独立重算，加原v26当前运行时32次复跑（旧31图结果逐项一致），共160调用。v27固定完整画面已知匹配76/106→81/106，v28为77/106；v28大裁块13枪头匹配，但整图10个框只有2个IoU匹配、固定val8枪头仍漏，局部训练拟合不能宣称系统改善。保持NOT_PROMOTED与生产4资产。

下一步中间尺度密集上下文训练支持及冻结连续片段局部CV诊断，同时继续剩余标注与所有原目标门禁。接续goal32-work-receipt.json、goal32-comparison.json、goal32-training-verification.json、iteration-summary-goal32.json、工作台review-goal32/snapshot-final.json。分支codex/rtx3050-device-delivery-20260904、HEAD 7db0dcd052891e0e189c5e931a27558256fea032和并行修改保留。Goal active/PROGRESS；质量PARTIAL_EVIDENCE，独立逐类、完整视频/机位/素材/播放、候选性能及语义正式归档NOT_PROVEN。


## goal33 接续更新

完成两个约95秒既有真实派生片段的FP v24固定、TP v23/v27对照，共3040源预测，逐帧去重/跟踪重算一致，恢复0调用，模型变更阻断复用。TP候选枪头预测跟踪前6323→后2293；实际7.875秒密集区15→4，另16.875秒手持移液枪检测漏框。框数不是GT计数；不能把全部跟踪丢弃当真实漏检。生成两个95秒有封面的诊断回放，文件面板queued，浏览器播放及主系统归档尚未验证。

全部1520推理像素唯一对应原视频原帧PTS；现有抽样时间与原帧PTS最多差62.151ms，估算索引也不等于实际索引。独立映射已完成，运行时原帧身份传播与跨机位物理同步仍待验收，不能用此偏差直接归因所有机位错配。下一步优先修正源帧身份链路、核对密集目标跟踪保留和手持工具漏检，再继续规范标注和固定对照训练。

当前382图3352框全部记录保持、51完整训练不变，无新增训练或生产替换。继续分角色检测、核对关联后互补证据；统一/双模型公平对照未完成。接续运行根goal33-work-receipt.json、candidate-video-goal33/receipt.json、video-review-goal33/pts-proof-attempt2/receipt.json及工作台review-goal33/snapshot-final.json。分支codex/rtx3050-device-delivery-20260904、HEAD 7db0dcd052891e0e189c5e931a27558256fea032及并行修改保持。Goal active/PROGRESS，质量PARTIAL_EVIDENCE、完整验收NOT_PROVEN。


## goal34 接续更新

原帧身份已贯通普通/持久分片FFmpeg抽帧、虚拟时间转换、模型输入包和FrameEvidence.source_frame；记录原始PTS/时间基、物理来源、像素SHA与缺失/歧义状态，抽样时间保持独立。末帧保持显式标识；旧身份检查点拒绝复用且保留旧文件；中途失败不再混入从头回退解码。上游新showinfo无包位置时立即记录未知，不能外推本机4.4.2的成功到所有平台。

同两段约95秒真实派生片段，4次对照共6080实际模型源帧预测（唯一1520画面），全部6080去重/跟踪独立重算通过；取证三次4560身份观察匹配goal33独立原帧表，最终代码1520全部匹配。每次恢复0调用/原账本摘要不变，4生产资产保持。首次22.067秒、关闭取证诊断16.016秒、优化19.827秒、最终19.717秒，完整性能门禁未通过。像素及时间网格相同，不同冷启动的预测仍有细小数值和少量选框/轨迹差异；关闭取证也出现，原因待隔离，不能声称逐位一致或质量已改善。

164相关检查、全src/tests的ruff和compileall通过。工作台382图3352框全部记录、51完整训练保持，本轮无新训练。接续goal34-work-receipt.json、source-frame-goal34/runtime-verification-final.json、candidate-video-goal34-final/receipt.json及工作台review-goal34/snapshot-final.json。下一步让派生素材/播放/正式归档消费原帧身份，继续查询性能与冷启动差异排查、剩余标注训练、独立逐类、角色/统一模型对照和完整真实链路。分支codex/rtx3050-device-delivery-20260904、HEAD 7db0dcd052891e0e189c5e931a27558256fea032及并行修改保留。Goal active/PROGRESS；本机指定原帧账本PROVEN，总体质量PARTIAL_EVIDENCE、完整验收NOT_PROVEN。


## goal35：关键素材消费原帧身份与架构取舍（2026-09-08）

继续采用“分人称检测，关联后互补证据”的开发基线；统一类别与标注规范，各角色训练及评测。统一模型、分角色模型、共享特征加角色分支的实际优劣仍NOT_PROVEN，须按同一来源隔离的验证集和明确训练/运行预算对照。新第三人称原始数据已收到，不能再记录为等待提供。架构取舍不能掩盖手持目标漏检、密集跟踪丢弃或错选架上器材；一致类别预测不能确认同一物体或动作。

本轮在source_frames.py加入有界原帧重取：核对机位/角色、唯一物理文件和分段映射、大小/mtime、原始时间基、解码PTS及包位置，并要求BGR像素SHA与推理账本完全一致。CPU简单滤镜按原PTS选择，仅读取附近2.1秒输入且20秒子进程超时；输出采用账本输入分辨率，当前真实例为960宽。不同解码器/缩放方式若像素不一致则保留unverified，未验证所有CUDA缩放及其他FFmpeg版本。

关键帧物化优先使用核验通过的原帧，原有派生实验片段继续作为关键短片来源。近似取帧退路仍可输出画面，但不借用附近检测框、不把seek目标写成实际解码时间。原图侧车升级/2，记录JPEG摘要、原帧证明和经对齐变换的实际时间；角色/拼图侧车保存frame_sources，短片侧车不冒充完整原帧证明。语义复核只有在/2图像摘要和原帧证明匹配时才附加已选关键帧，明确物理跨机位同步仍未验证。缺失原帧证明的原始画面、短片和时间序列可继续保留；所有媒体解码失败的跨事件隔离仍需后续处理。

本机已有两路约95秒Pipetting派生视频与goal34-final真实模型账本，实际抽查16个分散时间点（每角色8个，含0、7.875、13.125、16.875、27.625、47.5、70、94.875秒）原帧PTS/包位置/像素全部一致。另用明确provisional的诊断事件执行真实素材物化，生成两角色关键帧及拼图、两角色短片及拼接短片，保留正式未准入状态。自动选取7.5秒网格对应FP7529.412ms、TP7541.899ms原帧，这两个时间不证明两机位物理同步。实际查看TP16.875秒账本框和诊断拼图：手持部分遮挡枪仍漏检，参与物仍选到架上的枪；因此只证明框与原帧绑定，不能称质量问题已解决。

第一次诊断未启用派生缓存，却要求复跑禁止重编码，断言失败；原媒体由现有失败回滚逻辑保留，日志/脚本与首次产出留存。独立attempt2补齐本地缓存配置后，首次物化1.553秒、复跑0.446秒；禁止重新生成短片的复跑通过，六媒体摘要逐字不变，三短片复用已验证缓存。仅局部素材耗时，不是端到端性能。原视频与旧账本摘要重新核对，模型调用0、训练0、标注修改0，工作台382图3352框与51次完整训练保持。

146项相关测试、ruff check src tests、compileall src tests通过；包含变帧率/B帧/非零起点/分段原帧重取、错误文件/角色/位置/像素拒绝、旧账本不借框、新侧车图像绑定及已有素材/复跑/部分交付契约。源码快照和阶段差异保存在本地运行根key-frame-material-goal35，真实证据入口attempt2/real-material-receipt.json、native-frame-checks.json、final-verification.json；旧goal34文件保留。分支codex/rtx3050-device-delivery-20260904与HEAD 7db0dcd052891e0e189c5e931a27558256fea032未变，保留并行未提交修改。

上述指定原帧及局部物化/复跑PROVEN；总体质量PARTIAL_EVIDENCE；独立逐类检测、架构公平对照、全部标注、物理机位关联、参与物选择、主系统浏览器真实播放、全媒体失败隔离、候选引擎性能、语义完整归档及稳定发布NOT_PROVEN。Goal active/PROGRESS，未替换生产模型；继续处理参与物与密集跟踪、补充来源标注及系统可见验收。

溯源复验补充：goal34回执的158个链接中152个原路径摘要不变；6个链接是本轮授权修改的实时源码/测试或追加文档，旧字节已在goal35/before按原摘要保留。首次将全部实时路径都要求不变的汇总断言失败，失败记录留存；分清历史快照与当前文件后158份旧证据均可核对，4生产资产摘要保持。


## goal36：素材故障隔离、选择性恢复与真实浏览器播放（2026-09-08）

上一轮goal35属PROGRESS；本轮复验其96个链接文件摘要保持。关键素材按机位、关键帧/短片、拼图/拼接视频分别尝试。预期读取、解码、编码失败返回明确失败记录，继续独立产出和后续事件；取消及未分类程序错误不伪装成成功。缺少拼接前提时记unavailable，不合成黑占位。失败输出移出当前事件引用，原文件保留；帧写入使用现有失败回滚保护。所有存活侧车在末尾统一更新完成状态和当前媒体引用，避免早期in_progress或旧映射继续充当最终状态。

运行回执记录每项完成/失败、retry_event_ids和逐次失败历史；选择性复跑合并未涉及事件的记录，清空当前已恢复缺口但保留历史。部分报告增加待补全素材与事件重试范围，溯源索引绑定物化回执。质量门禁要求媒体数量齐全且新物化状态completed；即使遗留六个旧别名也不能绕过缺口。动作准入状态不因生成文件自动提升，诊断事件始终provisional。六种确定性故障（单帧、单短片、整路、源选择、拼图、拼接视频）及不掩盖程序错误有覆盖。

在goal34-final真实模型账本及既有两路约95秒Pipetting视频上，实际注入首事件第一机位关键帧解码不可用。另一机位画面、三短片和第二事件六媒体继续生成，共10份；解除故障只复跑首事件，补齐12份，原10份摘要不变且三短片缓存复用、重新编码调用0。最终源码运行首次2.119秒、恢复0.434秒，范围仅诊断素材物化，非整体性能。首轮及最终轮回执均保存；最终轮还核对所有存活侧车的完成状态一致。真实来源/旧账本与4生产资产摘要保持，模型调用0、训练0、标注修改0。

实际浏览器检查发现：已有实例42195的候选播放器有封面，阶段素材卡和聚焦播放器却未设置poster，且preload=none。当前前端两处复用统一播放器，提供真实关键帧封面、明确播放按钮、失败提示/重试与单独打开入口；聚焦不强制自动播放，数据绑定保留倍速等控件。画面对照文案不再直接宣称同步。原实例保持，通过本地只读GET/HEAD预览服务展示当前前端，后端仍引用既有Local-User-Acceptance-20260908阶段结果，不是新检测候选的部署验收。

系统Chrome152.0.7977.82实际点击“播放关键片段”后，视频readyState=4、1280×360、时长5.866667秒，连续两次播放时间前进，页面错误0；源码SHA与浏览器实际加载app.js一致。已实看播放前封面及播放中截图，后者出现变化的手/试管架内容。最初的异步页面初始截图、键盘/原生坐标播放尝试未证明播放成功，均留存；显式play()诊断后，以最终可见按钮点击完成实际UI核验。界面仍有较大黑色留边，已有两路素材的动作对应仍待核准，不能将播放通过当机位或动作准确性。

385项相关测试、ruff check src tests、compileall src tests、node --check app.js通过。另有一项原有Web测试硬编码样式版本70，与当前index/styles版本71不符；该断言不读取本轮修改的app.js，本轮未修改index/styles，也未顺手修正该无关失败。首次新增poster依赖使一个独立JS测试缺少location模拟，补齐实际依赖后通过。原包/模型/源码状态、阶段差异与失败日志保留。

真实故障隔离/选择性恢复、指定浏览器封面/按钮播放PROVEN；总体模型质量PARTIAL_EVIDENCE。全部标注、独立逐类与架构对照、参与物/密集跟踪、物理机位对应、新候选完整视频/引擎性能、语义正式归档和稳定发布NOT_PROVEN。工作台仍382图3352框、51完整训练。下一轮继续剩余密集标注、手持/背景器材判别与跟踪质量，并将已验证的前后端变更纳入完整系统验收；不重复以局部产出替代最终指标。Goal active/PROGRESS，未替换生产资产。

接续：运行根material-resilience-goal36/final-real/real-resilience-receipt.json、browser-playback-receipt.json、final-verification.json与goal36-work-receipt.json。修复预览http://127.0.0.1:54517，进程746552（exec会话9884），只读代理原后端42195，浏览器15已打开；后续须核对进程实际存活，不依此记录推断。分支codex/rtx3050-device-delivery-20260904、HEAD 7db0dcd052891e0e189c5e931a27558256fea032保持，全部未提交及并行修改保留。


## goal37 接续：新增第三人称项目标注（2026-09-08）

T028/T029/T030已完成第二遍区域外逐件复核，版本3/2/2，新增83明确实例、保留9处未知；同20260525源组train。工作台382图3435框：完整复核88图、区域外复核64图、待复核18图、草稿161图、排除51图。新第三人称169原图已处理14图区域外、155图草稿。原图和其他379图未变，历史审计保持；新审计头f34f7e3d48a3a12527eb4b1071690236e3b1c4b692f92547b99de4d4f1ae5197。

下一批32图部分监督预检通过，沿用goal32的29图及原验证标签，仅加3训练图；全项目普通导出仍631阻断。本轮无新导出/增强/训练/模型调用，累计51次完整训练；4生产资产保持。工作台Chrome实际显示T030原图、29实物与3未知，页面错误0。首次列表按钮计数包含未知区域的检查脚本错误已修正，保留失败记录。goal36的214份历史文件摘要保持。

详情见MODEL-ITERATION-RESULTS-20260907.md的goal37段；接续运行根goal37-work-receipt.json及annotation-goal37、工作台review-goal37/snapshot-final.json和cohort-next.json。分支codex/rtx3050-device-delivery-20260904、HEAD 7db0dcd052891e0e189c5e931a27558256fea032保持；Goal active/PROGRESS，总体质量PARTIAL_EVIDENCE。继续补充多来源及手持移液枪/密集标签、训练/真实视频对照，统一模型与分角色模型的实际优劣仍NOT_PROVEN。


## goal38 接续：补标与两次候选迭代（2026-09-08）

T032–T036五图已初标及第二遍区域外复核，新增104实例、保留15未知；工作台382图3539框（完整88、区域外69、待复核18、草稿156、排除51），新第三人称169原图仍150草稿。与goal37三图合并新增8训练原图187实例，冻结37原图/175train文件/2val裁剪；旧标签和验证图片字节不变，32增强预览与导出一致，部分监督消费者校验通过，全项目普通导出仍626阻断。

实际完成TP v29/v30两次训练，共1056优化步/2100样本访问，累计53完整训练（FP23/TP30）。系统one2many下固定106已标整帧实例匹配：v27=81、v29=67、v30=85；2完整验证裁剪v30=24/27。v29新增图拟合高但旧能力退步，v30低学习率从v27微调恢复旧能力，却仍漏T032两支手持枪，整帧枪头0/8。训练脚本one2one分数另记，不能与系统分数混用。两个候选均未晋升，不因总分改善通过质量门禁。

v30实际复跑95秒第三人称派生视频760帧，全部输入像素/原帧身份与v27一致，1520行两候选跟踪独立重算一致；7个真实原帧和9幅对照/放大已复核。16.875秒手持枪仍漏检；7.875秒枪头原始/跟踪数量变化不等于真实召回。局部检测10.070秒、恢复0.189秒且0模型调用/账本摘要不变，非全链路性能。Chrome实际显示T032原图及29实物/3未知，页面错误0。

详情见MODEL-ITERATION-RESULTS-20260907.md的goal38段及本地运行根goal38-work-receipt.json、goal38-model-comparison.json、video-review-goal38/visual-review.json。接续工作台review-goal38/cohort.json和annotation-verification.json。下一轮停止小验证集扫参，补充不同姿态/遮挡/背景的训练来源实例，固定验证/测试并继续实际失败点复查。分人称检测+可靠关联后互补证据仍为基线；统一模型是否更好NOT_PROVEN。

分支codex/rtx3050-device-delivery-20260904、HEAD 7db0dcd052891e0e189c5e931a27558256fea032及全部并行未提交修改保持，4生产资产未替换，无运行源码改动和外部付费调用。Goal active/PROGRESS，总体质量PARTIAL_EVIDENCE；全部标注、独立逐类/架构对照、参与物/物理机位关联、候选引擎/全长视频、语义正式归档与稳定发布仍NOT_PROVEN。


## goal39 接续：手持器材逐件标注与来源隔离（2026-09-08）

T042/T043/T044/T054/T055/T071/T080七张第三人称原图已逐件初标并第二遍区域外复核，均版本2；新增303实例、保留28未知，其中70支枪（7手持、63架上）、61管体、19管盖、49枪头（46盒内、3附着）。T055只有1根可辨管体与1个盖片段，不复制相邻图被手遮住的试管；同图每支枪分别取可见部分。实际查看31个原图/细节视图、24个初标叠框和11个修正叠框，第二遍重看全部7张未叠框原图，修正57个坐标边界。

工作台382图3842框：完整88图1212框、区域外76图2046框、待复核18图560框、草稿149图7框、排除51图17框。新第三人称169原图为26图区域外754框、143草稿。其他375图与全部source字段保持；1013条审计保留，新头aba2844ee582ce90eea6bef1220102860414fcc6d919db8a52608c0101453b51。

本轮7图统一保留third-original-unmapped-7509853b、unassigned和used_by_baseline，仅记录项目视觉第三人称。没有证据将其绑定原始视频/实验/人员/物理机位；6处CustomFlow源视频稀疏预览未建立映射，CSV的2026-06-18来源时间不能移植给未知原图。全项目普通导出626阻断，7图outside_ignore检查22阻断；来源未知时不进入训练。原goal38批次因审计头变化被契约拒绝；逐条确认原37成员版本/来源/标注摘要不变后，另存cohort-existing37-refrozen.json仅更新审计头，预检通过，7张新图不在其中。旧批次拒绝及检查脚本解析空stdout的失败记录留存，未修改检查契约。

Chrome实际逐页显示7图1920×1080原图、版本2、第三人称、未分组和38/37/36/41/52/37/62实例，错误0；T042/T055页面截图已实看。打开用户窗口T055的请求仅queued，不当作窗口已切换证明。goal38的366份历史证据文件及4个生产资产摘要保持。本轮没有新模型调用/训练/增强/训练导出，累计53次完整候选训练，生产未替换。

接续：工作台review-goal39/snapshot-final.json、visual-review-receipt.json、annotation-verification.json、cohort-existing37-refrozen.json；本地运行根annotation-goal39及goal39-work-receipt.json。下一轮继续来源映射与跨组近重复核查，补充有可靠来源的手持/密集真实图，再训练与实测；不继续对同一小验证集扫参。分人称检测+统一类别+可靠关联后互补证据为当前方案，架构质量优劣仍NOT_PROVEN。

分支codex/rtx3050-device-delivery-20260904、HEAD 7db0dcd052891e0e189c5e931a27558256fea032保持，本轮仅项目标注与进度文档变更，未提交修改及并行工作保留。标注保存/指定浏览器展示与来源阻断PROVEN；总体标注与模型质量PARTIAL_EVIDENCE；全部标注、独立逐类/架构比较、物理机位/关键素材质量、候选完整引擎性能、语义正式归档和稳定发布仍NOT_PROVEN。Goal active/PROGRESS。


## goal40 接续：六张明确来源组原图（2026-09-09）

T007/T009/T010/T011/T012/T013已初标及第二遍区域外复核，均版本2，新增106实例/14未知，20260519同组train及used_by_baseline保持。T011仅4根明确管体，T009只标可辨红盖、隐藏瓶身留未知。全库382图3948框（完整88、区域外82、待复核18、草稿143、排除51）；第三人称原始169图中137仍草稿。其他376记录、全部source及旧审计保持，1025条审计头ec85f81f1b87b6ead2aa502ba24e912da5796b227e4fe2f89ebcbda2d3ff5ea8。

cohort-next43.json沿用逐项核验不变的旧37成员及验证标签，仅追加六train，outside_ignore预检0阻断；全项目普通导出620阻断。本轮无新导出/增强/训练/模型调用，累计53次完整训练。Chrome六页实际显示原图/版本/角色/分区/数量，错误0；T009/T013截图实看。206份goal39历史文件和4生产资产摘要保持。

rgb_sel/rgb_topup仍未找到可信源视频映射，EXIF无拍摄时间，RAR成员修改日期不是采集日期。仅盘点USB-Imports与桌面62视频路径，未全盘/NAS排查；未知组继续unassigned，来源目录澄清待答。继续分人称检测+统一本体+可靠对应后互补证据，统一/双模型优劣未证明。

接续工作台review-goal40/snapshot-final.json、cohort-next43.json、annotation-verification.json；运行根annotation-goal40及goal40-work-receipt.json。后续继续可独立进行的规范标注、来源隔离与真实失败点验证，不因语义账户或部分未知来源停止其他阶段。分支codex/rtx3050-device-delivery-20260904、HEAD 7db0dcd052891e0e189c5e931a27558256fea032保持，未提交与并行修改保留。Goal active/PROGRESS，总体质量PARTIAL_EVIDENCE，完整验收NOT_PROVEN。


## goal41 接续：九张原图复核及52成员批次（2026-09-09）

T014、T016–T023 九张原图已逐件初标，保存版本1后重新实际看完整原图、全图和管架叠框完成第二遍，最终版本2、reviewed_partial/all_visible_outside_ignore。新增222实例、保留26未知；包括82管体、21管盖、10移液枪、19烧杯、9药匙、9手套掌指、5裸手。T023逐支核对十枪，普通短袖不标实验服；T022杯内过曝不补猜搅拌子。第二遍修正13个实物边界、9个未知边界及1个截断属性。9整图+31细节初查、9原图+3额外细节复查、19初标叠框和11修正叠框均实际查看；未查看的生成图在清单明确为false。

全库382图4170框：完整复核88图1212框，区域外91图2374框，待复核18图560框，草稿134图7框，排除51图17框。第三人称原始169图中41图区域外1082框、128草稿。其他373记录、全部source字段和旧审计前缀不变；审计1043条，头c8192996672718658cb6436e25b321d8f9348c85973eaa84e726f0fc0260e3c9。

逐项验证旧43成员版本/来源/标注摘要保持，只追加九个同组train成员，形成review-goal41/cohort-next52.json；52成员outside_ignore检查0阻断，原2张完整验证裁剪不变。原20260519/20260525组及used_by_baseline保持，不增加独立实验或独立验证。普通整库导出仍611阻断；本轮无训练导出、增强或模型调用，累计仍53次完整候选训练。

三份本地第三人称视频的18个稀疏预览与来源材料仍未建立rgb_sel/rgb_topup原图到视频的身份映射；未知组继续unassigned，未按文件修改时间推断采集时间。只说明所选时刻未建立映射，不声称全视频无手持枪或已经全盘排查。九张新增标注可补持杯、持瓶、药匙及背景覆盖，不能冒充手持移液枪额外训练来源已解决。

Chrome九页实际解码1920×1080原图，显示版本2、third_person、train及19/26/24/24/25/25/24/24/31实例，页面错误0，API与保存记录完全对应；T022/T023截图实际查看。向用户窗口打开T023请求仅queued，不作为已切换窗口证明。145份goal40历史文件及4生产资产重新哈希保持。

PROVEN：本轮逐件项目标注、保存复核、来源保留、浏览器展示和指定批次预检。PARTIAL_EVIDENCE：总体标注覆盖及项目视觉角色。NOT_PROVEN：全部原图标注、未映射来源/人员/近重复隔离、双模型与统一模型公平对照、完整逐类及真实长视频质量、物理机位/动作和关键素材准确性、候选引擎全长性能、付费语义正式溯源归档与稳定发布。保留当前分支/SHA、未提交及并行工作；Goal active/PROGRESS。

接续工作台review-goal41/snapshot-final.json、annotation-verification.json、visual-review-receipt.json、cohort-next52.json及运行根annotation-goal41、goal41-work-receipt.json。下一步对冻结52成员做有溯源的增强预览/导出，再使用固定验证和真实手持失效帧检验有明确目的的候选；同时继续未完成原图与源视频映射。保持未解决来源不入训练，候选质量未通过不替换生产。


## goal42 进展（2026-09-09）

已对 goal40/41 新增的15张已复核训练原图逐张查看增强对照，60份新增强与正式导出字节一致。冻结52个父图记录：50 train父图（含既有裁剪）、2张完整val裁剪；250训练图片=50父图+200离线增强。15新增原图共328已标实例，仍属于已有20260519/20260525组，不增加独立拍摄来源；原37成员标注及旧媒体/标签、验证集哈希不变。消费者outside_ignore/v2校验通过，整库普通导出仍611阻断。工作台382图4170框、88完整/91区域外/18待复核/134草稿/51排除，审计未修改。

实际完成第三人称v31（third-person-expanded-sources-v31），从v27初始化，沿用v30的学习率0.0001、freeze10、4轮：500优化步、1000样本访问、30.574秒，累计54次完整训练（FP23/TP31）。本轮同时增加数据量及更新步数，不能当成同预算因果对照。系统one2many实际调用v30best及v31best/last，共165张次；v30前40份预测与历史解析结果完全一致。

新增15张训练图的已知实例匹配261/328→311/328，其中管体78/116→107/116；这是训练拟合。2张完整val裁剪24/27→25/27，3张整帧已知实例仍85/106，但移液枪26/26→25/26（下降3.85个百分点），超过2个百分点保留门禁。gun-7缺少可独立分配的检测框；与其重叠的0.735框更匹配相邻gun-6，不能重复计数。旧T032两支手持枪、整帧验证8个枪头仍未匹配。训练内one2one结果单独记录为24TP/3FP/3FN，不与系统分数混用。v31未晋升，生产资产保持。

v31实际复跑95秒第三人称视频760帧，与v30全部输入像素/原帧身份一致；1520行双候选跟踪重算一致。检测10.465秒，恢复0.195秒、0模型调用及账本哈希不变；这是局部CV运行，非全链路性能。已实际查看7个原帧对照、2个细节：16.875秒手持枪仍在检测阶段漏检，v31同一手出现额外重叠框；7.875秒密集枪头预测增多伴随重叠/错类，不能据预测数量称召回改善。另实看T017训练管架改善和gun-7退步两幅细节对照；未改验证标注或阈值。

接续证据：运行根goal42-model-comparison.json、goal42-work-receipt.json、diagnostics-goal42、candidate-video-goal42、video-review-goal42；工作台review-goal42/cohort.json、export-validation.json，导出exports/tp-expanded-sources-augmented-goal42。保留分支codex/rtx3050-device-delivery-20260904、HEAD 7db0dcd052891e0e189c5e931a27558256fea032及并行未提交工作。本轮无产品源码改动、生产权重替换或付费语义调用；训练/指定实际调用PROVEN，系统质量PARTIAL_EVIDENCE，完整独立质量/物理机位关联/候选引擎全长性能/真实语义归档和稳定验收NOT_PROVEN，Goal active/PROGRESS。

下一步继续128张第三人称原图草稿及可靠来源映射，优先补充手持枪姿态/遮挡和密集实例训练来源；未知RGB组保持unassigned。需要冻结失败集分析单件分离、重复抑制和置信度，不能只对小验证集继续扫参数。按角色检测、统一类别和可追溯证据格式，只有来源/时间/实物关系可靠时互补；统一与分角色架构优劣仍无公平实验定论。


## goal43：失败阶段定位与长操作来源标注（2026-09-09）

实际截获v30/v31在2张既有图片和8个原生视频帧上的20张次系统预测，保留置信度过滤前张量、实际NMS输入及保留索引；按原4帧视频批次、960/float32/one2many、confidence=0.25/NMS=0.7执行，20份系统框均可从保存数据精确重建。T032左右手持枪两版均无IoU>=0.5且分数过0.25的同类候选；16.875秒手持工具诊断区域最高枪分数v30=0.00364、v31=0.00538（区域查询不是真值匹配），该漏检在NMS/跟踪前已经发生，单改去重无法解决。

密集验证图anchor7426分数0.3865→0.4683，但与较高分anchor7185的实际NMS重叠从0.698664跨至0.710051，超过0.7而被删除。原图、项目gun6/gun7框及两版候选已同屏实看；两候选都更接近gun6，橙框未覆盖项目gun7的最右端。NMS导致候选消失PROVEN，但v30额外框是否正确区分另一物体仍NOT_PROVEN；此前26/26仅为既定IoU分配结果，不能提升为逐支物理正确。未改验证标注或生产阈值，没有通过放宽去重宣称质量修复。

T123–T149、T151–T168共45张640×480长操作原图均实际查看并分类第三人称、train；保留既有复杂长操作来源组和used_by_baseline，不以组名first_person覆盖图片角色，不把跨视角算独立来源。T161/T168另完成逐件初标、保存后第二遍原图/叠框复核，最终版本3、reviewed_partial/all_visible_outside_ignore：新增23+22=45实例、4+4=8未知区域，包含每图4支独立可辨架上枪、2只手套手和1件药匙；第二遍修正7个框边界。其余43张只分类仍草稿。新增图并未补齐手持移液枪姿态多样性。

全库382图4215框：完整88图1212框、区域外93图2419框、待复核18图560框、草稿132图7框、排除51图17框。第三人称原始169图中43区域外、126草稿；角色86第三人称、83未知。全部source与其他337记录不变；新增49条审计，1092条头ac9f7f48fd1977b73315990871bd815c65bca47e898d720f415707ce32a02acf。原52成员版本/来源/标注摘要与2张完整val裁剪保持，追加两train形成cohort-next54，outside_ignore检查0阻断；普通整库导出仍523阻断。新批次尚未增强、导出或训练，累计完整训练仍54次（FP23/TP31），生产资产保持。

真实Chrome核验T123版本1草稿、T161/T168版本3区域外复核，均third_person/train、640×480原图，0/23/22实例及0/4/4未知，DOM/API与保存快照一致，页面错误0；两复核页截图已实看，未声称用户窗口已切换。实际查看清单区分生成但未查看的两幅最终upper叠框及T123页面截图。保存复核/指定浏览器展示与阶段定位PROVEN；全库标注及模型质量PARTIAL_EVIDENCE；全部数据标注、来源独立性、架构公平对照、实物/机位对应、关键素材及全链路质量、真实语义正式归档和稳定发布NOT_PROVEN。

当前路线为分别训练第一/第三人称检测器、统一类别和证据格式，来源/时间/实物关系可靠时做证据互补。双模型优于统一模型的结论仍需固定来源隔离数据与相同训练预算对照；单路缺失保留已成功产出，不凭另一机位补造本机位证据。下一步继续已知训练组逐件标注，并扩充有可靠来源的手持枪/密集目标失效样本；不把未映射RGB图投入训练，不在同一小验证集继续扫阈值。

接续：工作台review-goal43/snapshot-final.json、annotation-verification.json、instance-visual-review.json、role-visual-review.json、cohort-next54.json、browser-check.json；本地运行根failure-diagnostics-goal43、annotation-goal43及goal43-work-receipt.json。VisionCortex分支codex/rtx3050-device-delivery-20260904、HEAD 7db0dcd052891e0e189c5e931a27558256fea032保持；AnnotationWorkbench分支codex/annotation-workbench、HEAD f78875b210379469fb8934712b6fb9f77b8b6690保持。本轮未修改产品源码、未提交/推送或替换生产资产，保留并行未提交工作。Goal active/PROGRESS。


## goal44：裸手与未穿戴手套场景补标（2026-09-09）

T123/T129/T136/T143/T151/T158六张第三人称640×480原图已完成逐件初标及保存后的第二遍区域外复核，均版本3、reviewed_partial/all_visible_outside_ignore、train。分别24/25/26/17/21/20实例，共133新实例，包括36支移液枪、6只裸手、21个瓶盖与10个枪头盒；保留35未知区域。蓝色未穿戴手套和桌布不标为另一只手，空称量盘不标纸张；T143运动模糊及右侧截断区域继续保留未知，不按邻帧件数补画。六张均沿用既有复杂长操作组和used_by_baseline，不增加独立实验，也没有解决手持移液枪训练姿态不足。

实际查看6个全图、14个细节；初标版本2保存后在6幅原图/叠框并排图中重新检查完整原始像素，各图再检查枪架叠框，四张有手图另查上区叠框，共16幅初标叠框；修正10处实物边界及1项天平遮挡，7幅修正叠框已实际查看。参考同组T122/T150既有项目框保持类别口径，每张仍独立看图、修正状态与边界，未复制已复核状态；生成但未查看图明确列出。全部source、其他376记录及审计前缀不变。新增12条审计，共1104条，头461a0e384485a1c8da8e4eb16ec4bd11dbcb49c26972d086f0116b98e445a73d。

全库382图4348框：完整88图1212框、区域外99图2552框、待复核18图560框、草稿126图7框、排除51图17框；第三人称原始169图中49区域外、120草稿。逐项验证旧54批次成员版本/来源/标注摘要及2张完整val裁剪不变，只追加六train，cohort-next60的outside_ignore检查0阻断；普通整库导出529阻断。新批次未增强/导出/训练，累计完整训练仍54次（FP23/TP31），未启动模型或改变生产权重。

真实Chrome六页显示版本3、第三人称、train、640×480及全部实例/未知数量；DOM/API与最终快照一致，页面错误0。T136/T151浏览器截图实际查看；未声称用户窗口已经切换。接续工作台review-goal44/snapshot-final.json、annotation-verification.json、visual-review-receipt.json、cohort-next60.json、browser-check.json，以及本地运行根annotation-goal44、goal44-work-receipt.json。继续37张已知复杂长操作组草稿和其他未完成原图，补充可靠来源的手持枪及密集场景；未知来源不投入训练。新增标注可能支持手部背景误检诊断，改善效果仍需后续实际模型/视频对照。

本轮逐件项目标注、部分监督批次检查及指定浏览器展示PROVEN；总体数据覆盖与模型质量PARTIAL_EVIDENCE；完整数据标注、独立逐类/架构对照、物理机位和实物关联、关键素材质量、候选引擎全长性能、真实语义正式归档及稳定发布NOT_PROVEN。分支codex/rtx3050-device-delivery-20260904、HEAD 7db0dcd052891e0e189c5e931a27558256fea032保持；AnnotationWorkbench为codex/annotation-workbench、HEAD f78875b210379469fb8934712b6fb9f77b8b6690。无产品源码变更/提交/推送，保留并行未提交工作。Goal active/PROGRESS。


## goal45：手套、药匙和密集枪架逐帧复核（2026-09-09）

T124/T126/T132/T138/T146/T154/T162/T166八张640×480原图完成初标和保存后的第二遍区域外复核，均版本3、第三人称、train、reviewed_partial/all_visible_outside_ignore。新增185实例（48支移液枪、3只裸手、4只戴手套的手、3件药匙等），保留45处未知区域。未穿戴手套不标为手，天平后的用品盒不并入手部；T154仅见左腕/前臂而掌指重遮挡，保持未知。T162药匙柄左界由173修至166。初标实际查看8全图、16细节；保存版本2后再看8幅完整原图/叠框并排图及各图密集叠框、6幅上区叠框，并重看T162天平原图放大；修改后上区叠框已查看。查看清单和未查看生成图见visual-review-receipt.json。

全部source字段、其他374记录及历史审计前缀不变；审计1120条，头009d6757ea62fff3e8bbbbcd813e64546ec621924eac649a18e9a2d7da1a9f75。全库382图4533框：完整88图1212框、区域外107图2737框、待复核18图560框、草稿118图7框、排除51图17框。第三人称原始169图中57区域外、112草稿；已知复杂长操作组还剩29草稿。这8张仍属已有来源组，不增加独立实验。只读检查另发现三张6月16日保留测试图片目前仅手部初标；实际查看前两张，第三张未查看，全部保持test和原版本，未送入训练。

旧60成员及2张完整val裁剪保持，追加8张train得到cohort-next68（66 train父图含已有裁剪、2 val裁剪），outside_ignore检查0阻断。普通全库导出537项阻断，仍不能按普通YOLO丢弃ignore导出；此数是合同阻断数，不是检测错误数。本轮未导出/增强/训练，累计训练54次（FP23/TP31），未启动模型。真实Chrome八页版本/图片/实例/未知区域数量与快照和API一致，页面错误0；T154/T162截图实际查看，浏览器访问前后快照一致。未改变用户当前窗口。

架构建议维持分视角检测、统一类别与可追溯证据，跨视角互补必须满足实际来源、时间及对象关联。两个独立权重、共享骨干分视角头、单一混合模型应在相同来源隔离、训练预算、输入尺寸与真实视频评价下比较；当前没有公平架构对照，不能宣布双模型一定更优。单视角有效产出应可保存，缺失另一视角或语义服务时保留待补证状态及复跑入口。

接续证据：/home/x1/.local/share/annotation-workbench/review-goal45中的snapshot-final.json、annotation-verification.json、visual-review-receipt.json、cohort-next68.json、browser-check.json；运行根annotation-goal45/final-verification.json和goal45-work-receipt.json。指定标注与浏览器检查PROVEN；总体数据覆盖和现有模型质量PARTIAL_EVIDENCE；完整标注、独立逐类/架构对照、物理机位关联、真实关键素材质量、候选引擎全长性能、真实语义正式归档及稳定发布NOT_PROVEN。继续已知来源原图逐件复核及可靠来源的手持枪姿态补充；不以这些静态密集样本代替手持姿态证据。

VisionCortex分支codex/rtx3050-device-delivery-20260904、HEAD 7db0dcd052891e0e189c5e931a27558256fea032；AnnotationWorkbench分支codex/annotation-workbench、HEAD f78875b210379469fb8934712b6fb9f77b8b6690保持。无产品源码变更、提交/推送或生产资产替换，保留并行未提交工作。Goal active/PROGRESS。


## goal46：剩余后段操作原图逐件复核（2026-09-09）

T155/T156/T157/T159/T160/T163/T164/T165/T167九张640×480第三人称原图完成初标及保存后的第二遍区域外复核，均版本3、train、reviewed_partial/all_visible_outside_ignore。新增196实例（36支移液枪、8只裸手、8只戴手套的手、4件药匙等），保留50处未知区域。初标实际看9全图、18细节；保存版本2后重看9幅完整原图/叠框并排图、9密集叠框、9上区叠框，并另看T163/T164/T165/T167天平原图3倍细节，将四张称量纸框收紧为[209,159,301,245]，四幅修正叠框已实际查看。T167药匙位于开口瓶上方，双手、瓶盖、搅拌子与纸包按本帧遮挡重新标注；松散未穿戴手套、门把手、用品盒印刷和粉末不增加手或工具实例。

source-similarity.json记录9张RGB像素摘要均不同。T159/T160平均绝对RGB像素差约2.31/255，视觉高度相似；这只是像素描述，不能当语义独立性指标。所有图片保持已有复杂长操作来源组和used_by_baseline，不增加独立实验，不进入val/test。全部source、其他373记录和审计前缀不变；新增18条审计，共1138条，头1a8d4c93cb9c58bbae9f4aa33e8f26da1d098343c00b44aab054e799b565a5d4。

全库382图4729框：完整88图1212框、区域外116图2933框、待复核18图560框、草稿109图7框、排除51图17框。第三人称原始169图中66区域外、103草稿，已知复杂长操作组剩20草稿。旧68批次成员来源/版本/标注摘要和2张完整val裁剪保持，追加9张train得到cohort-next77（75 train父图含已有裁剪、2 val裁剪），outside_ignore检查0阻断。普通全库YOLO导出546项合同阻断，仍不得删除ignore导出。本轮未增强、导出、训练或启动模型，累计训练仍54次（FP23/TP31）。

真实Chrome九页版本、角色、split、640×480图像和全部实例/未知区域数量与最终快照及API一致，页面错误0；T155/T167截图实际查看，浏览器访问前后快照一致，未切换用户窗口。证据位于/home/x1/.local/share/annotation-workbench/review-goal46/的annotation-verification.json、visual-review-receipt.json、source-similarity.json、cohort-next77.json、browser-check.json，以及运行根annotation-goal46/final-verification.json和goal46-work-receipt.json。

本批逐件标注、区域外批次检查及指定浏览器页面PROVEN；总体数据覆盖和既有模型质量PARTIAL_EVIDENCE；完整数据标注、新样本模型收益、独立逐类/架构对照、物理跨视角关联、关键素材质量、候选引擎全长性能、真实语义正式归档及稳定发布NOT_PROVEN。继续20张已知来源原图及其余未完成数据，仍需补充可靠来源的手持移液枪姿态；静态枪架和持药匙不代替持枪证据。VisionCortex分支codex/rtx3050-device-delivery-20260904、HEAD 7db0dcd052891e0e189c5e931a27558256fea032；AnnotationWorkbench分支codex/annotation-workbench、HEAD f78875b210379469fb8934712b6fb9f77b8b6690保持。无产品源码变更、提交/推送或生产资产替换，保留并行未提交工作。Goal active/PROGRESS。


## goal47：玻璃后手部、松散手套与画面偏移逐帧复核（2026-09-09）

T125/T127/T128/T130/T131/T133/T134/T135/T137/T139十张640×480第三人称原图完成初标与保存后的第二遍区域外复核，均版本3、train、reviewed_partial/all_visible_outside_ignore。新增246实例，包括76支架上移液枪、11只裸手、1件实验服、1件药匙；55处未知区域保留。九张逐支可对应八支枪，画面偏移的T139仅可靠对应前四支，不按邻帧补件数。T125/T131透过天平玻璃可见的掌指与用品盒印刷/反光分开；松散未穿戴手套不标戴手套的手。T137第二遍6倍原始掌指细节检查后，将左手框左界62收至87以排除前臂，保留上缘可见拇指片段，修正叠框已实看。

初标实际看10完整原图、19细节；保存版本2后重看10幅完整原图/叠框并排图、10密集叠框、8上区叠框，另看1幅手部原始细节及1幅最终修正叠框。59幅不同查看图与生成但未查看图分别列于visual-review-receipt.json；两张浏览器截图另计。所有source、其余372记录和审计前缀不变，新增20条审计，总1158条，头5e7c94afa0fa79adcf5eff7af8a288feaf2e0d60307930822011f04895508917。十张都属于既有复杂长操作train来源和used_by_baseline，不增加独立实验。

全库382图4975框：完整88图1212框、区域外126图3179框、待复核18图560框、草稿99图7框、排除51图17框。第三人称原始169图中76区域外、93草稿；其中已知复杂长操作组剩10草稿，另83张来源未映射RGB仍不能投入训练。旧77成员的来源/版本/标注摘要与2张完整val裁剪保持；追加10张train得到cohort-next87（85 train父图含已有裁剪、2 val裁剪），outside_ignore检查0阻断。普通全库YOLO导出556项合同阻断，不能删除ignore绕过。本轮未增强、导出、训练或启动模型，累计完整训练仍54次（FP23/TP31）。

真实Chrome十页原图640×480、版本3、第三人称/train及逐件实例/未知数量与API和最终快照一致，页面错误0；T131/T139截图已实际查看，浏览器访问前后快照不变，未切换用户窗口。接续文件位于/home/x1/.local/share/annotation-workbench/review-goal47/的annotation-verification.json、visual-review-receipt.json、cohort-next87.json、check-next87.json、browser-check.json；本地运行根annotation-goal47/final-verification.json与goal47-work-receipt.json关联源文件、历史证据和生产资产摘要。

本批标注、特定部分监督批次检查与指定浏览器展示PROVEN；总体数据覆盖和既有模型质量PARTIAL_EVIDENCE；新样本模型收益、完整数据标注、独立逐类/统一与分视角架构对照、物理跨视角对应、关键素材质量、候选引擎全长性能、真实语义正式归档和稳定发布NOT_PROVEN。继续剩余10张已知来源原图、保留测试视频全类别复核及可靠来源的手持移液枪样本；静态枪架样本不能代替手持姿态证据。架构建议保持分视角检测、统一标签与来源/时间/实物关系可靠后的证据互补，优势仍需公平对照，单路缺失应保留已有产出和复跑入口。

VisionCortex分支codex/rtx3050-device-delivery-20260904、HEAD 7db0dcd052891e0e189c5e931a27558256fea032；AnnotationWorkbench分支codex/annotation-workbench、HEAD f78875b210379469fb8934712b6fb9f77b8b6690保持。无产品源码变更、提交/推送或生产资产替换，保留并行未提交工作。Goal active/PROGRESS。


## goal48：已知来源第三人称原图完成区域外复核（2026-09-09）

剩余T140/T141/T142/T144/T145/T147/T148/T149/T152/T153十张640×480原图完成初标及保存后的第二遍区域外复核，新增204实例（48支移液枪、4只裸手、3件实验服、1件药匙等），保留54处未知区域。T144按偏移后的枪架位置重新生成5倍细节；T140/T141各八支可辨枪，其余各前四支，出画和模糊片段不按邻帧补件数。T152/T153只框天平后可辨掌指，松散手套和裸腕不扩大为手。T149长柄和勺部可对应药匙，其他不清楚金属片段仍未知。

实际查看10全图、22初标细节；版本2保存后逐张重看10幅完整原图/叠框并排图、10枪架叠框、10上区叠框，另看两幅4倍用品盒及三幅6倍盒层原始细节，修正8个框可见边界，5幅修正叠框已实际查看。72幅不同查看图与生成但未查看图分别记录，两张浏览器截图另计。初标版本2、几何复核版本3均保留；发现说明误带上一批55处未知总数后，经正常CLI版本检查另存版本4，删除逐图说明中的该批次总数措辞。实际本批54处；版本4仅改说明，几何、未知区域、人称、分区和复核状态与版本3完全一致，未伪造或覆盖历史版本。

十张最终均版本4、third_person/train、reviewed_partial/all_visible_outside_ignore；全部source、其他372记录、旧87批次成员与历史审计前缀不变。新增30条审计，共1188条，头7d501b9a17c14d0cb293aa69614bb54a38d1084843aaa32062c8ef6efbea21e7。169张第三人称原始图中86张已完成区域外复核；剩余83张RGB原图来源未映射仍草稿，未进入训练。已知复杂长操作组不再有原图草稿，但局部未知仍在，不能称所有原图完整标注。

全库382图5179框：完整88图1212框、区域外136图3383框、待复核18图560框、草稿89图7框、排除51图17框。追加10张同组train形成cohort-next97（95 train父图含既有裁剪、2张既有完整val裁剪），outside_ignore检查0阻断；普通全库YOLO导出566项合同阻断，不能删除ignore绕过。本轮没有增强、导出、训练或模型启动，本目标已记录完成训练仍54次（FP23/TP31）；这些新增同组图片不增加独立实验或证明模型收益。

真实Chrome十页均核验版本4、角色/分区、640×480原图和逐件/未知数量，DOM/API与最终快照一致，页面错误0；T144/T153截图已实际查看，浏览器访问前后快照不变，未切换用户窗口。接续证据为/home/x1/.local/share/annotation-workbench/review-goal48/的annotation-verification.json、visual-review-receipt.json、review-notes-correction.json、cohort-next97.json、check-next97.json、browser-check.json、remaining-non-todo-originals.json，以及本地运行根annotation-goal48/final-verification.json和goal48-work-receipt.json。

下一步优先继续6张6月16日保留测试视频图（第一/第三人称各3张）的全类别标注和复核，保持test及同源隔离，不用于候选训练；继续6月17日验证图和F119/F120/F125待复核项，不能以两张小val裁剪代替全场景质量。完成必要标签/冻结后，再按已验证部分监督契约预览增强、导出和训练对照；继续补足可靠来源的手持枪姿态，保留83张未映射RGB来源问题。本批逐件区域外复核、批次检查及指定浏览器页面PROVEN；总体数据/模型质量PARTIAL_EVIDENCE；完整原图标注、新批次模型收益、独立逐类及架构比较、物理跨视角关联、关键素材质量、候选引擎全长性能、真实语义正式归档和稳定发布NOT_PROVEN。

VisionCortex仍为codex/rtx3050-device-delivery-20260904、HEAD 7db0dcd052891e0e189c5e931a27558256fea032；AnnotationWorkbench仍为codex/annotation-workbench、HEAD f78875b210379469fb8934712b6fb9f77b8b6690。无产品源码变更、提交/推送或生产资产替换，保留并行未提交工作。Goal active/PROGRESS。


## goal49：六张保留测试帧逐件补标与真实解码核验（2026-09-09）

本轮为实际进展，完整目标保持 active。对 2026-06-16 同一实验的 `C-Tube-Place-And-Dilute-view-02-side-b-{0,1,2}` 与 `view-03-front-{0,1,2}` 六张原有 test 帧进行了初标、保存后重新查看完整原图、密集区及第二遍复核。六张共 233 实例，相比原手部草稿净增 226；保留 41 处未解决区域，全部仍为 needs_review/partial/test，不增加完整 reviewed 或完整测试资格。第三人称三张为版本3；第一人称0/2为版本3、1为版本4（最后统一容器总框包含已连接瓶盖的边界）。第二遍修正17条框/未知区记录，另有第一人称1的一项容器边界修正。三张第一人称的瓶口专门放大仍不能确认独立蓝盖，撤回该假定，保留一般样品瓶和瓶口未知区域；不武断宣称已确认开口。右缘清楚可见的红盖容器及盖另行补标。

当前项目 382 张 / 5405 实例：完整复核 88/1212，区域外复核 136/3383，待复核 24/793，草稿 83/0，排除 51/17（斜线前为图片数、后为实例数）。83张原始第三人称未映射来源仍未进入训练。原有97项批次的每项图片、版本、来源和标注摘要均未变，仅冻结到当前审计头；其中95项train父图（含原有裁剪）、2张固定完整val裁剪，没有本轮test帧。其 outside_ignore 检查0阻断；普通全项目检查572阻断（退出2为仍有不完整数据的预期门禁）。当前审计1201条，头 `d0f16525406aa3c6b427c901e8213cadd3dba380059160be911813cec0faa958`。其他376条图片记录和全部来源字段不变。

PROVEN：68份实际查看的唯一原图/查看图/叠图；初标保存后六张完整原图均再次打开检查。真实Chrome六页逐项核对修订、角色、test分区、尺寸、实例及未知区数，页面错误0；第三人称1与第一人称1截图实际查看，浏览器后快照未变。两条已整理的本地视频及证据包SHA再次核验，CPU FFmpeg实际解码在三个采样时刻附近的帧，六张工作台原图均存在唯一逐像素完全一致的候选帧（仅指所检查的小时间窗口）。这排除了本次PNG抽帧额外引入色彩变化，第一人称过亮已存在于已整理的源片段；原始相机录制层仍待核对。

精确匹配帧PTS：第三人称为 `1096000/119`、`3656000/119`、`6224000/119` 毫秒；第一人称为 `655200/71`、`2184000/71`、`3712800/71` 毫秒。相对于声明采样毫秒，第三人称有约16.9、33.3、-16.5毫秒差异。保留整数PTS/time_base及像素比较回执，不重写冻结的旧来源记录；跨机位证据关联须使用真实帧时间，不能把请求采样时间冒充解码PTS。此处只证明相对于现有片段的帧身份，不能证明物理安装或绝对跨机位同步。

本机FFmpeg 4.4.2首次拒绝辅助核验命令的`-fps_mode`，进程明确退出1；已记录选项探测失败并改用兼容的`-vsync 0`完成解码，没有安装依赖或改动产品代码。没有新增训练/模型启动/付费调用，累计完成的训练仍54次，生产权重与引擎保持原摘要。

PARTIAL_EVIDENCE：项目标注覆盖及既有模型质量。NOT_PROVEN：这六张的全类别完整测试指标、独立泛化、原始相机曝光成因、绝对机位同步、两角色方案公平比较、全长性能及真实语义全链路正式归档/稳定发布。方舟恢复门禁仍保留，但局部产出已保存并可继续复核。

下一步：将本轮发现的瓶口/瓶盖歧义纳入既有冻结训练图的针对性核对；修正若发生则重新冻结版本。继续原有验证帧复核，同时用已合格批次推进候选训练和原固定评测，单独列出未完成test和局部val的证据范围，不以其中一项缺失丢弃其他产出、不按分数选图或移动保留分区。源片段实际PTS差异纳入后续真实机位/材料溯源核验。

回执：`/srv/sentinel-data/VisionCortex3090Ti/Runtime/Project-Detector-Pilot-20260907/goal49-work-receipt.json`；逐件版本/审计、六页浏览器及精确帧比较：`/home/x1/.local/share/annotation-workbench/review-goal49/`。本轮保留 VisionCortex 分支 `codex/rtx3050-device-delivery-20260904`，HEAD `7db0dcd052891e0e189c5e931a27558256fea032`；AnnotationWorkbench HEAD `f78875b210379469fb8934712b6fb9f77b8b6690`。未提交修改和既有证据保留。


## goal50 接续：第三人称v32完成训练，真实漏检仍未通过（2026-09-09）

已针对性核对47个现有蓝盖瓶/盖标签，无需修改；全库382图5405框、1201审计及全部来源/分区保持。97项合格批次冻结为95train父图+380增强、2固定val裁剪；新增45张同组原图1009实例，12张/48变体实际预览，部分监督v2检查通过，普通全库572阻断。旧52记录及504媒体/标签保持；未纳入83张未知来源原图和未完成test。

实际完成TP v32四轮952更新/1900访问，累计55次训练（FP23/TP32）。固定系统诊断300源图调用：新增45训练图966/1009→1001/1009；2val裁剪25/27→24/27；3整帧85/106保持，实验服和防护用品盒退步。枪类25/26→26/26仍含相邻实例重叠框的贪心匹配，不能当不同实物找回。训练图T032右手枪找回，左手仍漏。

真实95秒派生视频760帧完成，10.645秒，恢复0.221秒/0推理/账本不变；760原生帧身份与像素对照v31一致，1520跟踪行重算一致。实际查看7固定整图+2局部以及3静态诊断对照：16.875秒手持枪仍漏，密集枪头重叠错类仍在，不晋升候选。生产模型/引擎和产品源码保持；没有新浏览器、付费语义或正式归档验收。

详情见MODEL-ITERATION-RESULTS-20260907.md的goal50及本地运行根goal50-work-receipt.json、goal50-model-comparison.json、diagnostics-goal50、candidate-video-goal50、video-review-goal50；工作台review-goal50保留全部快照、批次、增强检查及失败后准备重试记录。分支codex/rtx3050-device-delivery-20260904、HEAD 7db0dcd052891e0e189c5e931a27558256fea032保持；Goal active/PROGRESS，质量PARTIAL_EVIDENCE。继续同数据/等更新的来源平衡对照与真实失败点，补可靠手持目标来源；分角色检测+可靠关联后证据互补仍为当前路线，统一/共享/双模型的公平优劣NOT_PROVEN。


## goal51 接续：固定来源平衡及裁剪均未解决手持漏检（2026-09-09）

同goal50的97项/475train文件完成TP v33来源平衡对照：4epoch、952更新、1900访问，累计56次训练（FP23/TP33）。实际300静态源预测及95秒视频760帧；控制记录、760原帧身份/像素、1520跟踪行重算和0推理恢复通过。两张val裁剪24/27→25/27，但整帧已知匹配85/106→84/106、手套手3/5→2/5；手持枪与密集目标错误仍在，不晋升。

固定手部上下文策略另做131裁剪预测，7个完整包含的已知枪框定位支持由整图6降至裁剪0；其余436个未完整包含的已知枪框不计漏检。实际看4幅裁剪和T031/T032/T033原图，确认类型依据后保持原标签；裁剪策略不接入系统。全库382图5405实例、1201审计与来源分区保持，批次0/普通全库572合同阻断。未增加浏览器验收、生产替换、付费调用或产品源码变更。

详见MODEL-ITERATION-RESULTS-20260907.md的goal51及运行根goal51-work-receipt.json、goal51-model-comparison.json、hand-probe-goal51/localized-gun-diagnostic-v2.json。下一步补已知train来源的手持工具尺度/遮挡样本并双遍复核，保留全图与ignore；不再扫小验证集比例/ROI，不将保留组转训练。继续分角色检测和可靠关联后证据互补；统一/共享/双模型的公平优劣NOT_PROVEN。Goal active/PROGRESS，质量PARTIAL_EVIDENCE；物理机位、独立质量、全长性能、方舟真实语义正式归档仍待验收。主仓库原分支/HEAD 7db0dcd052891e0e189c5e931a27558256fea032与工作台原分支/HEAD f78875b210379469fb8934712b6fb9f77b8b6690保持。


## goal52 接续：手持训练裁剪完成，v34未通过，工作台视窗已修复（2026-09-09）

T031/T032已有train原图新增4张来源绑定裁剪、58逐件框，原图/密集细节实际查看及保存后第二遍复核，均版本2完整reviewed；不是4个新独立实验。全库386图5463框：完整92/1270、区域外136/3383、待复核24/793、草稿83/0、排除51/17。原382条记录/来源不变，审计1213条。批次101项=99train父图+2固定val；16新增强实际预览，495train文件含396增强，旧97项及954媒体/标签字节保持，批次0/普通全库572合同阻断。

TP v34实际4epoch、992更新、1980访问，累计57次训练（FP23/TP34）。312静态源预测及95秒视频760帧完成；同输入像素、跟踪重算及0推理恢复通过。新增训练裁剪已知匹配7/58→43/58，但整帧验证85/106→82/106、手套手3/5→1/5；16.875秒手持枪仍漏，密集错框仍在，不晋升。新增数据也增加更新次数，不声称等算力因果提升。

工作台fit仅按宽度缩放导致竖图底部不可见，已改annotation_workbench/web/app.js同时适配宽高。真实Chrome复现旧问题并检查4张图/两种窗口尺寸/3档缩放、原始框和指针映射，页面错误0、快照未变；26项相关测试及node语法检查通过。此项PROVEN只覆盖工作台显示，主系统视频/机位质量仍待验收。

详细结果见VisionCortex的docs/MODEL-ITERATION-RESULTS-20260907.md之goal52；运行根goal52-work-receipt.json、goal52-model-comparison.json和工作台review-goal52保存完整证据。下一步先核对训练收敛和整图上下文，再冻结有依据的迭代；继续验证/test复核及未知原图来源。分视角检测、可靠关联后证据互补仍为当前路线，公平架构优势NOT_PROVEN。质量PARTIAL_EVIDENCE；物理机位、独立质量、全长引擎性能、真实语义正式归档仍未通过，Goal active/PROGRESS。原两仓库分支/HEAD保持，无提交/生产替换/付费调用；保留并行未提交工作。


## goal53 接续：20轮训练补足拟合，真实手持漏检仍在（2026-09-09）

同goal52的495train文件完成TP v35固定20轮，实际4960更新/9900访问，累计58次训练（FP23/TP35）。预先固定last20对v34last4，不以原生best或系统分数事后选权重。发现本机原生best及CSV损失均基于one2one，而系统使用one2many；原生E2ELoss还逐轮降低one2many权重。源码及推导系数已归档，尚非实测分支损失遥测。

实际312静态源预测、两个最终权重各760视频原帧。训练裁剪44/58→56/58，6个重复枪标签3→6匹配；T032训练整图左右手枪均检出。整帧局部已知83/106→84/106，但实验服1/3→0/3、枪26/26→25/26、管盖10/10→9/10；两张完整val裁剪多漏一枪。真实16.875秒手持枪仍漏，密集错框仍在，不晋升。760输入身份/像素对照、1520跟踪重算及0推理恢复通过；5静态、7视频整图、2细节、1训练曲线均实际查看。

全库386图5463框、1213审计、全部来源/角色/分区及101项批次保持；批次0/普通全库572阻断，无新增标注、增强、生产替换或产品源码修改。详情见VisionCortex的docs/MODEL-ITERATION-RESULTS-20260907.md之goal53及运行根goal53-work-receipt.json、goal53-model-comparison.json、annotation-goal53/branch-loss-schedule-audit.json。下一步先加入真实分支损失/权重遥测，再验证针对系统分支的训练方案；继续未完成验证/test及未知来源核对，不把保留视频转训练。Goal active/PROGRESS，质量PARTIAL_EVIDENCE，物理机位/独立质量/全长性能/真实语义正式归档仍NOT_PROVEN。两仓库原分支/HEAD与未提交修改保持。


## 2026-09-09 goal54：实际分支损失记录与原生训练对照

本阶段为 **PROGRESS**，总 goal 继续 active；质量仍为 **PARTIAL_EVIDENCE**，候选不晋升。分支 `codex/rtx3050-device-delivery-20260904` / SHA `7db0dcd052891e0e189c5e931a27558256fea032` 保持，既有未提交修改保留。

显式 `--trace-branch-loss` 已实现并通过 84 项定向检查、`ruff check src tests` 和 `compileall`。默认关闭，沿用原生优化目标；实际训练逐批记录两分支损失、权重及加权结果，绑定预处理输入、每轮快照和恢复检查。原生 CSV 的损失仅反映 one2one，不能替代系统实际使用的 one2many 损失。

实际新增 TP v36 `third-person-branch-observed-v36`：与 v35 相同 495 训练文件（99 父图＋396 离线增强）、v27 初始化、冻结10层、960/FP32/batch2、学习率 .0001、固定20轮。完成4,960次更新、9,900图片访问，耗时 267.840891 秒。累计59次训练（FP23/TP36），不把训练次数当质量。每轮分支记录已重算，已完成回执恢复检查未改写产物。

本次 **PROVEN**：best/last 各708个模型张量分别与 v35 完全相同，采样顺序和除耗时外的 CSV 相同；固定104图的 best/last 系统预测分别精确复现。权重文件含不同运行元数据，文件哈希不同，不能称二进制相同。v36预定 last SHA `d75c63a6d2b5005fefeb4518f22d756d980efbaba8b446d7e1088470ab6d2d1c`；原生 best 仍在第7轮，独立保留，不按系统分数选替代终点。

真实视频新增760帧，与 goal53 已冻结的 v35last760帧比较，全部源像素/帧身份一致，重新核对1,520行跟踪计算；复跑0次源图推理且账本不变。视频输出数组非精确相同：31,075原始框/27,272跟踪框按同类位置对应，最小IoU .991248，最大坐标差 .561962源像素，最大置信度差 .000288874；9帧有18个框换序，对应后跟踪编号变化0。各帧类别数量一致；数值差的精确来源 **NOT_PROVEN**，不宣称跨进程逐位一致。

实看5张静态对照、7个视频时间点、2处放大和1张分支曲线。16.875秒手持白色移液枪仍漏检，7.875秒密集枪头仍有重复和试管/管盖混淆，实验服和过大容器/架体框问题仍在。整图已知匹配84/106，其中实验服0/3、移液枪25/26、试管盖9/10；两张完整验证裁剪23TP/1FP/4FN，均无质量提升。未将局部标注匹配解释为完整视频准确率。

386图/5,463框、1,213审计条目及标签版本/来源/分区未变；训练批次检查0阻断，全项目仍572阻断。未新增独立来源，未完成全部标签；数据保持项目标注和 `independent_ground_truth=false`。未新做浏览器、真实语义正式归档、物理机位同步或候选引擎全原视频性能验收；这些门禁仍 **NOT_PROVEN**。未调用付费模型、未替换生产模型或引擎。

回执：`/srv/sentinel-data/VisionCortex3090Ti/Runtime/Project-Detector-Pilot-20260907/goal54-work-receipt.json`；比较：`/srv/sentinel-data/VisionCortex3090Ti/Runtime/Project-Detector-Pilot-20260907/goal54-model-comparison.json`；代码/测试及真实损失证据：`/srv/sentinel-data/VisionCortex3090Ti/Runtime/Project-Detector-Pilot-20260907/annotation-goal54`。下一步预先固定一个与系统 one2many 分支一致的目标对照，保留原生默认与固定终点；同时继续补齐来源映射、完整验证和困难样本。损失权重递减本身不证明是漏检原因，不盲目延长轮数、调阈值或把6月17日诊断视频放入训练。


## 2026-09-09 goal55：固定一对多目标的真实训练与视频对照

本轮 **PROGRESS**，完整 goal 保持 active；模型质量 **PARTIAL_EVIDENCE**，v37 不晋升。分支 `codex/rtx3050-device-delivery-20260904` / SHA `7db0dcd052891e0e189c5e931a27558256fea032` 保持；原有未提交修改、生产权重及引擎保留。

新增显式 `--branch-loss-policy one2many`，默认 native 不变。仅允许区域外监督、分支记录、detector范围及patience=0；实际一对多损失直接返回，另一分支前向值保留记录但不进入反向传播，避免“乘零”仍触发闲置参数权重衰减。策略与逐批权重、预处理输入和每轮快照/恢复核对绑定；混用或缺失拒绝。95项定向检查、ruff src/tests和compileall通过；原v36真实完成回执在新代码下复验不变。

真实训练 TP v37 `third-person-one2many-objective-v37`：与v36同v27初始化、495输入（99父图＋396增强）、960/FP32/batch2、freeze10、LR .0001、固定20轮。完成4,960次更新、9,900图片访问，247.876737秒；累计60次（FP23/TP37）。逐轮实测权重均one2many=1/one2one=0，采样顺序与v36一致。最终one2one头66个保存参数张量与初始化相同，其余300个参数张量中201个变化；不宣称未优化头的运行统计量或共用特征被冻结。

第20轮一对多定位/分类损失分别.203850/.175562，比原生对照.216798/.184866稍低，不能等同泛化收益。主终点事先固定last，SHA `e72d22e0ddad967466ad51d5ba5e70260eed4aba0443b2642271590e5bd6168c`。原生best落在第1轮，因为它仍按未作为训练目标的one2one验证；保留辅助诊断，不因其小裁剪分数更好而换终点。

实际系统104图×3权重共312次源预测/源NMS；控制组完整复现v36。两张完整验证裁剪从23TP/1FP/4FN到24TP/1FP/3FN，但三张局部已标整图从84/106到82/106：戴手套的手4/5→3/5，移液枪25/26→24/26；实验服0/3、单支枪头0/8、搅拌子0/3仍未匹配。前两类已知匹配下降分别.20/.03846，超过.02保留目标；手类同时不足20例。这里只是项目已知实例对照，不是完整逐类召回。

候选实际运行同一95秒真实派生片的760源帧/760次NMS，与goal54冻结v36视频比较；全部源像素/帧身份一致，重新核对1,520行去重/跟踪计算。复跑0次源图预测且账本不变，更换模型身份在启动前被拒绝。v37原始/去重后/跟踪框31,033/31,021/27,798；v36为31,075/31,062/27,272。单支枪头跟踪框2,636→3,180是预测次数，不能当正确实例召回改善。

实际复看5静态细节、7固定视频时间点、2放大和1训练曲线。T032训练图持枪分数上升，但16.875秒真实视频白色手持移液枪仍漏检；7.875秒密集枪头仍重叠并覆盖空孔，试管/管盖混淆、袖部误检为过大容器及架体误报仍在。此配方未解决关键错例。

386图/5,463框、1,213审计条目及标签/来源/分区不变；批次0阻断，全项目572阻断。83张未映射原图及完整验证/测试标签仍未完成；本轮没有新增独立来源或标注。未做新浏览器、物理机位对应、候选引擎全原视频性能、付费语义与正式归档验收；这些以及稳定发布仍 **NOT_PROVEN**。真实训练/明确目标消费、指定源帧身份和复跑为 **PROVEN**，不能扩大到完整目标已完成。

下一步优先可靠训练来源中持枪姿态、遮挡、尺度、清晰度与完整验证标签。先审查实物像素尺寸和现有标签，再决定经逐图复看的增强；不继续盲扫分支权重/轮数，不改阈值，不把6月17日诊断视频或邻帧放入训练。对不能辨清的区域保留未知，不因需达标而补猜标签。

结果：`/srv/sentinel-data/VisionCortex3090Ti/Runtime/Project-Detector-Pilot-20260907/goal55-model-comparison.json`；回执：`/srv/sentinel-data/VisionCortex3090Ti/Runtime/Project-Detector-Pilot-20260907/goal55-work-receipt.json`；代码/检查/实测损失证据：`/srv/sentinel-data/VisionCortex3090Ti/Runtime/Project-Detector-Pilot-20260907/annotation-goal55`。


## 2026-09-09 goal56：三张第一人称验证原图逐件复核

本轮 **PROGRESS**，总 goal 保持 active。先核验 goal55 回执关联的306份文件均与摘要一致、当前工作台快照未变，再继续原有 F119/F120/F125 的待复核标注。这3张1920×1080原图均来自原有 test_long_0512_camera01 组，保持 first_person/val；原始相机安装及同组映射的既有限制未被改写，不宣称独立验证。没有新增训练或模型启动。

实际逐图查看原图、密集管架3/4倍放大及重点区域，第一遍保存后重新打开完整原图和逐类叠框做第二遍检查。修正天平可见总边界、遮挡和截断；F119重新确认左缘蓝盖并修正样品瓶类别、补盖框，撤回下缘失焦皮肤的裸手候选并保留未知；F125第二遍将延伸至右缘的打开玻璃门纳入同一天平。前排管、可辨后排管身、管盖、架体分别核对；其余透明管身仍无法可靠分离，保留未知，不能以孔数补猜。三张均为版本4，分别22/22/26实例、2/1/1未知区，状态仅为区域外已复核，仍不具备完整val资格，未转入训练。

当前386图/5463实例：完整复核92图1270框，区域外139图3453框，待复核21图723框，草稿83图0框，排除51图17框；其中16张为既有派生裁剪。本轮总框数不变，1个裸手框撤回、补1个可见蓝盖框，不把改框数量当新图。新增6条审计，共1219条，头46b72c2ee4759a54a379f04ba859804b659fea53930d3e2701c43ab7943c1839。其他383条图片记录、全部来源/角色/分区、历史审计前缀保持。原101项批次逐项版本/来源/标注摘要核对不变后重新冻结审计头；批次检查0阻断，全项目普通导出仍572阻断。

真实Chrome检查三页版本、角色、val分区、1920×1080媒体解码、实例/未知数量及API完整记录一致，页面错误0；F119/F125截图实际查看，浏览器前后快照一致。共实际查看24份唯一原图/查看图/叠图/页面截图，三张完整原图均在第一遍保存后重新打开。无产品源码改动、新训练、数据增强、付费语义或正式归档；累计训练仍60次（FP23/TP37），最新v37的真实视频问题仍未解决，不晋升。四份生产权重/引擎及v27对照权重摘要再次核验保持。

本轮指定标注修订、审计保持和三页浏览器可见性 **PROVEN**；整体数据/模型质量 **PARTIAL_EVIDENCE**。完整标注、独立逐类评测、分人称优于统一模型、物理跨机位对应、关键素材质量、候选引擎全长性能、真实语义正式归档及稳定发布仍 **NOT_PROVEN**。继续补齐原验证/test标签、核对83张未映射原图来源，并从可靠训练来源补足手持器材/遮挡/尺度样本后再冻结训练；不以局部复核或重复训练代替全场景验收。

证据：`/home/x1/.local/share/annotation-workbench/review-goal56/`中的annotation-verification.json、visual-review-receipt.json、cohort-check.json、full-check.json、browser-check.json及snapshot-final.json；阶段回执为`/srv/sentinel-data/VisionCortex3090Ti/Runtime/Project-Detector-Pilot-20260907/goal56-work-receipt.json`。VisionCortex分支codex/rtx3050-device-delivery-20260904、HEAD 7db0dcd052891e0e189c5e931a27558256fea032，AnnotationWorkbench分支codex/annotation-workbench、HEAD f78875b210379469fb8934712b6fb9f77b8b6690均保留；未提交/推送，保留已有未提交工作。


## 2026-09-09 goal57：参数核对与生产权重同条件真实评测

本轮PROGRESS，总goal active。用户询问参数、与现有权重差距及何时更好后，读取四份权重保存的训练参数，并补做原生产.pt与FP v24 best / TP v37 last的同条件RoleScanner one2many比较；固定960/FP32/batch1、预测底线.001、报告.25、NMS .7、匹配IoU .5、max_det1000，无额外去重或跟踪。15张完整FP val和TP2完整val裁剪+3部分整图的版本、来源、标签均与旧冻结记录一致，没有按分数换图。两次实际运行各40源图调用；第二次纠正Ultralytics因不存在的settings父目录回退/tmp的局部运行设置，并断言本地目录，40份原始输出全部精确复现；首次证据保留。

PROVEN仅限固定范围：FP生产147TP/12FP/65FN→v24 171/11/41，P/R92.45%/69.34%→93.96%/80.66%，mAP50/95 77.33%/68.77%→90.01%/75.15%；TP两完整裁剪生产15/7/12→v37last24/1/3，P/R68.18%/55.56%→96.00%/88.89%，mAP50/95 56.51%/40.30%→95.05%/77.09%。共有21类也提高：FP147/203→164/203，TP15/26→23/26。3张部分TP整图已知60/106→82/106，不算完整P/R。独立按原始预测重算整数计数与评估报告一致，候选micro复现旧系统分支结果。实际查看FP容器和搅拌子两处新增漏检以及TP原固定整图对照；密集错框仍可见。FP容器9/11→8/11、搅拌子5/6→4/6，管盖仍2/8；不晋升。

当前同为YOLO26s，21→23类；原保存配置120epochs、MuSGD、lr.001、FP batch32/TP16、AMP开、无freeze；当前FP v24实跑80epochs/batch4，TP v37实跑20/batch2，AdamW/lr.0001/freeze10/FP32。原120仅配置值，不是已核验完成轮数；FP v24是累计第23次完整运行，旧v23缺完整回执。FP96训练父图/15val、来源平衡；TP99父图+396离线增强/2val、逐图均匀。v37从v27初始化，只优化one2many，必须把初始化对照与生产对照分开。旧权重记录框架8.4.51、当前8.4.28；不声称单个参数解释效果变化。

本轮没有训练、标签变更、引擎重建、生产替换或付费调用，仍386图/5463框、1219审计、60完整训练。整体质量PARTIAL_EVIDENCE；全场景独立逐类、实际部署引擎、完整视频机位/素材及语义正式归档仍NOT_PROVEN，不能给出可信上线日期。下一步补齐完整验证、已知关键退步及手持困难训练来源，未来每轮同时列生产权重和初始化权重对照。参数与结果详见docs/MODEL-PARAMETERS-AND-BASELINE-COMPARISON-20260909.md；本地diagnostics-goal57-localsettings保存运行账本、完整参数、原预测、独立重算和实际查看回执，goal57-work-receipt.json为接续。主仓库原分支/HEAD保留，源码和生产资产未改，未提交/推送。


## 2026-09-09 goal58：原始画面逐件标注与旧场景退步对照

本轮为 PROGRESS，总 goal active；整体质量 PARTIAL_EVIDENCE，候选不晋升。实际查看原始第三人称包按 frame 文件名排序的首3张未标图 T000/T001/T002，在任何本轮模型预测前完成初标与保存后的整图第二遍复核。3张均版本0→1→2，新增28/30/30共88实例；每张分别10支中下架枪、4支右架可辨枪、2枪架、3瓶盖。修正逐枪和架体边界、衣摆/皮肤/鞋的区分、天平被手套轻微遮挡；T001一只可辨裸手、T002两只裸手，未穿戴蓝手套不标为戴手套的手。右缘残片、前横杆下不能对应的端部、橙盖周边不明瓶体及失焦皮肤/衣料保留未知，3/4/4处。仅 reviewed_partial/all_visible_outside_ignore；未通过删未知成为完整图。

依据实际连续画面固定台面与操作者对面入镜，完成项目视觉 third_person 分类；保留原来源组 third-original-unmapped-7509853b、unassigned 及 used_by_baseline，不把自带Bottom view字样当物理机位证明。尚无可靠原实验/视频/人员映射，这3图没有进入训练或验证。没有新增独立来源。当前386图/5551框：完整92图1270框、区域外142图3541框、待复核21图723框、草稿80图0框、排除51图17框。新增6条审计，共1225；其他383条、全部源记录/原图字节和分区、历史审计前缀保持。原101项批次逐项版本/来源/标注摘要保持，重新冻结审计头后0阻断；全项目普通导出仍572阻断。

真实Chrome检查3页版本2、960×540媒体解码、中文类别/28/30/30实例和3/4/4未知区、角色/未分组及API一致，页面错误0；T000/T002截图实际查看。共27份唯一实际查看文件、30次查看（含每张原图保存后再次打开、密集分组叠框、浏览器和模型对照）。浏览器及推理前后快照一致。没有改产品源码。

在冻结标签上实际调用当前RoleScanner one2many：960/FP32/batch1、预测底线.001、报告.25、NMS .7、IoU .5、max_det1000，不额外去重。生产/v37及初始化v27/原生对照v36分别跑同3图，共12源调用/12源NMS。88个已标实例匹配：生产70、v27最佳38、v36末轮53、v37末轮54；移液枪分别39/42、15/42、30/42、30/42。生产天平3/3、裸手3/3、防护用品盒3/3，v37均0；v37瓶盖6/9、枪架4/6提高，不能掩盖其他类别退步。原预测匹配索引/同类/置信度/IoU/去重一一对应已重算；未知区域使完整FP/FN不可得，不报告整图精确率/召回率或独立泛化。两张生产/v37完整叠图实际查看：候选把手套盒内手套误当戴手套的手，漏掉天平/裸手和部分架上枪；生产也有把未穿戴手套误标为手的情况。

退步在v27初始化已存在，不能归因于v37的一对多目标本身。追溯实际训练记录，v27从21类生产初始化时已迁移12个分类输出权重/偏置并处理旧枪头盒语义；v36/v37从其23类权重继续，不是漏开21→23迁移。现99个TP父图所有23类均有已标正例，例如天平88、用品盒76、裸手48；数量存在不证明场景分布足够，也不能由此确定遗忘的唯一原因。下一轮保持99父图/495文件、固定验证和同等20轮日程，改从生产权重重新微调并显式做类别迁移，检查原场景保留、旧固定验证和真实视频；新3图仍隔离，不用模型分数改标签/分区。

本轮没有新训练，累计仍60次（FP23/TP37）；生产权重/引擎保持，无付费调用、引擎重建或全长视频/语义正式归档验收。完整数据、独立逐类、物理机位对应、素材质量、候选引擎全长性能和正式发布仍 NOT_PROVEN。VisionCortex分支 codex/rtx3050-device-delivery-20260904、HEAD 7db0dcd052891e0e189c5e931a27558256fea032；工作台分支 codex/annotation-workbench、HEAD f78875b210379469fb8934712b6fb9f77b8b6690 保持，未提交/推送，未提交工作保留。

证据：工作台数据目录 review-goal58 的 annotation-verification.json、visual-review-receipt.json、browser-check.json、training-coverage-and-lineage.json、diagnostic-verification.json；模型运行根 diagnostics-goal58 / diagnostics-goal58-checkpoints 的预检、原预测、账本和已知实例对照。阶段回执 goal58-work-receipt.json。

## goal59：从生产权重重启训练及同源真实视频复核（2026-09-09）

结论 **PARTIAL_EVIDENCE，v38不晋升**。累计61次完整训练（第一人称23、第三人称38）。本轮没有改动标注；全项目仍386图5551框、1225条审计，保留原分支和未提交工作。goal58新复核的T000/T001/T002仍为部分标注、unassigned，未进入训练或完整验证。

第三人称v38从原生产21类权重重新初始化，显式迁移21→23类分类输出（12个输出参数张量）。数据、增强、采样顺序及日程保持v37一致：99训练父图、495训练文件，960/FP32/batch2，AdamW学习率0.0001，余弦lrf .05，freeze10，20轮，实际4960次参数更新、9900次样本访问、244.74秒。one2many损失权重1、one2one权重0，末轮last是训练前固定终点。与v37采样记录逐字节一致。首次运行在GPU占用预检退出，未创建训练输出或启动模型；空闲后原计划重试完成，没有停止其他任务。模型last SHA为`61c054d99c37b08dc6a1b9c065a473111d3981c1098fec3905b212ee2c4cd42f`。

相同系统分支/尺寸/精度/报告阈值下，3种权重各实际推理107图，共321次源图模型调用及321次源NMS。旧生产与v37在goal57的5张TP对照图及goal58的3张原始图上逐项复现。

| 冻结范围 | 原生产 | v37-last | v38-last |
| --- | ---: | ---: | ---: |
| 2张完整验证裁剪/27实例，TP/FP/FN | 15/7/12 | 24/1/3 | 24/3/3 |
| 同上精确率/召回率 | 68.18%/55.56% | 96.00%/88.89% | 88.89%/88.89% |
| 同上mAP50–95 | 40.30% | 77.09% | 79.28% |
| 3张部分验证整图/106已知，匹配数 | 60 | 82 | 87 |
| T000/T001/T002的88已知，匹配数 | 70 | 54 | 72 |
| 同原始图移液枪42已知，匹配数 | 39 | 30 | 35 |
| 同原始图裸手3已知，匹配数 | 3 | 0 | 2 |
| 同原始图实验服2已知，匹配数 | 2 | 0 | 0 |

v38原始图的天平、防护用品盒各恢复3/3，但关键移液枪、裸手、实验服仍低于旧权重；小验证裁剪多2个误检。部分图只报告已知实例匹配，不计算整图精确率/召回率。2张裁剪mAP50虽为1.0，报告阈值下仍有3FP/3FN，不能称完美，更不能代表全场景。来源被旧基线接触且缺完整独立来源映射，所有这些都是项目证据，不是独立真值。

真实视频复跑固定的第三人称95秒片段：8fps，760帧、190批次，960/FP32/batch4，实际760次源NMS，检测运行8.99秒。760帧像素、网格与原生PTS/packet身份全部与v37对照一致；重算1520份两版本逐帧去重/跟踪阶段并核对实际输出。缓存复跑0模型调用、0NMS且账本不变；更换模型后的续跑在模型启动前被拒绝。这里只证明该固定单机位片段，不能推导全长多机位SLA或物理同步已通过。

实际查看7个固定时刻的原生帧对照、2个细节放大、1张裸原图及4张完整静态对照，共14个文件。**16.875秒的白色手持移液枪仍在原始检测中漏检，7.875秒密集枪头仍错框/重叠。**v38在部分时刻补上手与实验服，但末尾实验服仍不稳定。T032训练图两支手持枪已检出不能证明跨视频泛化；未将这段诊断视频加入训练，也未改标签去迎合预测。

下一步补充不同来源的手持、遮挡与密集器材训练样例，完成整图验证；继续保留生产/v38和这段视频作固定回归。原生产权重、引擎均未替换；第一人称本轮未新训。逐类P≥.90/R≥.85、关键类不退步、密集实例、真实多机位素材与性能/溯源归档等门禁继续适用。付费语义阶段仍缺账户恢复证据，总goal保持active。

证据位于运行根`/srv/sentinel-data/VisionCortex3090Ti/Runtime/Project-Detector-Pilot-20260907/`：`goal59-training-plan.json`、`third-person-production-restart-v38/experiment.json`、`diagnostics-goal59/`、`candidate-video-goal59/`、`video-review-goal59/`、`annotation-goal59/visual-review-receipt.json`、`goal59-model-comparison.json`及`goal59-work-receipt.json`。本轮未修改产品代码；并行任务的API/流水线等改动保留并记录，不把共享工作区当作整体冻结源，实际运行各自存档源码快照。

## goal60：原始图双遍复核与同条件保留检查（2026-09-09）

新增T081/T091/T102/T103/T112/T121六张原始图两遍复核，共157框（60个移液枪实例）及13个保留未知区域，全部版本2、third_person/unassigned；来源组及基线接触状态未变，不进入训练或完整验证。全项目386图5708框：92完整、148区域外、21待复核、74草稿、51排除，1237条审计。其余380条记录及原101项训练批次不变；重冻批次检查0阻断，普通整项目仍572阻断。浏览器6页核对成功、0页面错误。

实际21次源图推理/NMS，新6图157已知匹配：生产125、v37 124、v38 148；移液枪60/60恢复，但T112/T121手持纸相对生产漏检，纸张14/15→12/15。类别细分不确定的橙盖瓶体差异不当作实物误检。保留此前真实视频失败，不晋升候选；本轮未新训，累计仍61次。详见VisionCortex的MODEL-ITERATION-RESULTS-20260907.md中goal60，以及运行根goal60-work-receipt.json和标注数据目录review-goal60。总goal active，完整系统质量NOT_PROVEN；继续原图标注、来源核对和有可靠来源的困难样本/整图验证。

## goal61进展（2026-09-09）

PARTIAL_EVIDENCE；goal持续active。本轮新增3次同一95秒真实派生片段运行，共2280源帧；旧生产960与v38的960/1280/1536均漏掉固定16.875秒手持枪，分辨率提高不足以晋升。修复`source_frames.py`的原尺寸NV12/默认CPU颜色转换兼容：只允许一次额外CPU转换，每次仍严格核验PTS、packet与像素SHA；61项相关测试、ruff、compileall、diff检查及21次真实像素复现通过。未更新运行服务，不能声称用户页面黑屏全部修好。没有新训/新标注，仍386图5708框、累计61训练（FP23/TP38）；生产权重/配置保留。详细参数、对照图和限制见VisionCortex `docs/MODEL-ITERATION-RESULTS-20260907.md` 的goal61；机器回执`/srv/sentinel-data/VisionCortex3090Ti/Runtime/Project-Detector-Pilot-20260907/goal61-work-receipt.json`。全项目标注、手持/密集质量、真实原片多路全链路/正式归档仍NOT_PROVEN。
