## 2026-09-10 goal94：恢复标注训练主线，同系统分支核对召回与误检

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

# 当前训练参数与生产权重同条件对照（2026-09-09，goal64，最新训练v40）

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


**goal61更新：未新增训练，训练参数与goal59 v38一致。完成960/1280/1536真实片段推理对照；加大分辨率未修复固定手持漏检，生产配置不变。NV12证据帧重建兼容修复经61项测试/21次真实像素重建验证，浏览器部署仍待核验。**

当前第一人称仍为v24；第三人称最新实验为v38-last。v38恢复部分旧场景检出，但关键类退步与真实手持漏检尚未解决，**不能替换生产，系统质量为PARTIAL_EVIDENCE**。下面先列最新结果，其后保留goal57/58原始记录，历史表中的“当前”仅指当时版本。


## goal60新增原始场景对照（未新训，参数保持下表）

先完成六张原始图157实例的逐件初标和第二遍原图复核，再固定标签、相同系统参数对照生产/v37/v38；另有T000控制图逐项复现原预测，共21次源调用/NMS。新6图已知匹配125→124→148/157；60个移液枪实例分别60/43/60匹配，纸张15已知分别14/6/12匹配。v38在T112/T121新增明确的手持纸退步，之前95秒真实视频的手持白色移液枪漏检未被解决。标签有未知区域，来源映射未核实且被旧基线接触，仍未分区，不纳入训练/完整验证或独立评测，不报告整图精确率/召回率。

累计完整训练仍61次（FP23/TP38），当前项目标注增至386图5708框。原生产资产不替换，PARTIAL_EVIDENCE。最新标注与同条件原始预测见MODEL-ITERATION-RESULTS-20260907.md的goal60及goal60-work-receipt.json；以下goal59参数与历史结果保留其原冻结范围。

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

## 历史参数与同条件对照（goal57，保留原记录）

当前结论：在这批固定项目验证图上，第一人称v24和第三人称v37-last的总指标均高于原生产权重，证据为真实重新推理的同条件对照。**完整系统效果仍为PARTIAL_EVIDENCE，候选不晋升。**这不能证明新实验/新实验员泛化、完整视频关键素材质量或生产引擎性能；两张第三人称裁剪尤其不能代表整幅场景。

## 后续原始场景对照：发现明确退步（goal58）

上一节固定验证范围的改善仍成立，但新增原始图T000/T001/T002的88个已标实例上，生产权重匹配70个，当前v37仅54个；移液枪39/42→30/42，天平3/3→0/3、裸手3/3→0/3、防护用品盒3/3→0/3。这3张在本轮预测前完成逐件初标/第二遍复核，仍有未知区域且保持unassigned，不计完整验证或独立真值。

初始化v27为38/88、原生对照v36为53/88，因此退步早于本次分支目标修改。21→23类输出迁移已核对存在；不是所有旧类缺训练正例。下一轮从生产权重重启同数据/同日程的受控微调，并同时保留固定原验证、这些原始场景退步检查及真实视频。v37不替换生产；两张小验证裁剪不能代表全场景效果。完整过程见MODEL-ITERATION-RESULTS-20260907.md的goal58和运行根goal58-work-receipt.json。

## goal57时的训练参数（v37为当时版本）

旧参数直接读取两份原权重中的train_args；120是配置轮数，原权重epoch已剥离为-1，不能据此声称独立核验原训练实际跑满120轮。当前参数同时核对experiment.json、args.yaml、CSV及真实训练记录。

| 参数 | 原第一人称 | 原第三人称 | 当前第一人称 v24 | 当前第三人称 v37 |
| --- | --- | --- | --- | --- |
| 模型 | YOLO26s | YOLO26s | YOLO26s | YOLO26s |
| 类别数 | 21 | 21 | 23 | 23 |
| 保存参数量 | 9,964,118 | 9,964,118 | 9,965,666 | 9,965,666 |
| 权重所记录Ultralytics版本 | 8.4.51 | 8.4.51 | 8.4.28 | 8.4.28 |
| 配置epoch | 120 | 120 | 80（实跑80） | 20（实跑20） |
| 图像尺寸imgsz | 960 | 960 | 960 | 960 |
| batch | 32 | 16 | 4 | 2 |
| 优化器 | MuSGD（记录值musgd） | MuSGD（记录值musgd） | AdamW | AdamW |
| 初始学习率 | 0.001 | 0.001 | 0.0001 | 0.0001 |
| 余弦调度/lrf | 开/.01 | 开/.01 | 开/.05 | 开/.05 |
| weight_decay | .0005 | .0005 | .0005 | .0005 |
| freeze | null | null | 10个层编号范围 | 10个层编号范围 |
| AMP | 开 | 开 | 关，FP32 | 关，FP32 |
| nbs | 64 | 64 | 4 | 2 |
| warmup_epochs / bias_lr | 3 / .1 | 3 / .1 | 0 / 0 | 0 / 0 |
| patience | 0 | 0 | 0 | 0 |
| seed | 0 | 0 | 20260907 | 20260907 |
| 在线scale参数 | .33 | .5 | 0 | 0 |

当前两角色分别维护权重和数据，共用23类本体，新增移液枪架、枪头盒；旧11类的整盒语义与新单支枪头语义仍须按既有迁移记录分别解释。并未换成更大的骨干模型。旧与新框架版本不同是已知条件，不能把效果变化单独归因于优化器、批大小或版本。

第一人称v24从对应生产权重初始化：96张训练父图（含原有裁剪）、15张完整验证图/212实例，来源平衡有放回采样；没有离线增强，在线增强关闭。80轮实际1920次参数更新、7680样本访问，实验耗时189.83秒（含该实验评测）；末轮学习率5.03662e-6。版本编号v24但累计完整训练23次，旧v23缺完整实验回执，不能计作完成。

第三人称v37从候选v27最佳权重继续微调：99张训练父图加396张离线增强，共495训练文件；4个来源组，2张完整验证裁剪/27实例。均匀逐图采样，20轮4960次参数更新、9900样本访问，实验耗时247.88秒；末轮学习率5.5848e-6。实验固定优化one2many分支（权重1），one2one损失仅记录而不进入反向传播（权重0）；主比较终点事先固定last，不能见到小裁剪分数后换成第1轮的原生best。

两者的box/cls/dfl配置权重为7.5/.5/1.5。区域外监督保留未知区域，不将其作为可靠负样本；验证只使用完整标签。当前TP离线增强每父图4份，亮度.85–1.15、对比度.92–1.08、gamma .92–1.08、缩放.94–1.00、平移比例.03、模糊半径0–.35、JPEG质量88–98、最小框边4像素。增强不增加独立来源数；val/test不增强。

## 刚完成的同条件比较

两角色原生产.pt与固定候选分别经过当前RoleScanner同一路径：one2many、960、FP32、batch1、预测保留底线.001、报告阈值.25、原生NMS IoU .7、匹配IoU .5、max_det1000，无额外框去重/跟踪。类别映射保持原编号；21类旧模型与23类候选均按项目目标评价，另列共有21类结果。

| 角色与范围 | 权重 | 精确率 | 召回率 | TP / FP / FN | mAP50 | mAP50–95 |
| --- | --- | ---: | ---: | --- | ---: | ---: |
| 第一人称15张/212实例 | 生产 | 92.45% | 69.34% | 147 / 12 / 65 | 77.33% | 68.77% |
| 同上 | v24 | 93.96% | 80.66% | 171 / 11 / 41 | 90.01% | 75.15% |
| 第三人称2裁剪/27实例 | 生产 | 68.18% | 55.56% | 15 / 7 / 12 | 56.51% | 40.30% |
| 同上 | v37-last | 96.00% | 88.89% | 24 / 1 / 3 | 95.05% | 77.09% |

第一人称净增24个正确匹配、少1个误检，召回提升11.32个百分点；第三人称这两个小裁剪净增9个正确匹配、少6个误检，召回提升33.33个百分点。总体精确率/召回是micro，mAP是有支持类别的macro；不与训练日志最佳阈值混比。

只看共有21类，第一人称召回72.41%→80.79%（147/203→164/203），第三人称57.69%→88.46%（15/26→23/26）；改进并非仅由新增类别计入造成，但完整逐类门禁仍未满足。

另外同3张部分标注TP整图，106个已知实例匹配60→82；未解决区域使完整FP/FN未知，不能据此计算整图准确率或将其他预测全判误检。本轮未运行连续视频；goal55真实95秒视频仍有手持白色移液枪漏检和密集枪头/管盖混淆。

## 为什么现在仍不能替换

- FP通用容器9/11→8/11，磁力搅拌子5/6→4/6；这两处退步的具体原图预测已实际查看。试管盖仍2/8，试管30/36，不能用平均分掩盖。
- TP完整验证只有来自同一原图的两个裁剪，许多关键类别不足20正例或完全缺失。整图密集区仍可见错框，不能称全场景已通过。
- 旧训练器直接one2one指标（例如FP85.21%/67.92%）与本表one2many不同，不能跨表相减。FP v24与TP v37本轮结果分别复现既有系统分支固定验证的micro计数；本次补上的是生产权重在同一路径的对照。
- 这里是同条件PyTorch权重比较，尚未证明现有TensorRT部署配置与新候选引擎在全长、多机位真实视频上的效果和性能。原始数据被旧基线接触，项目标注不是独立真值。

## 何时可以说比现有权重好

“固定验证集的总体指标更好”本轮已有PROVEN证据；“可替换生产、在系统中稳定更好”仍NOT_PROVEN。至少需按角色/类别达到P≥.90、R≥.85、每类20个复核正例的既定门槛，解决关键类退步、密集重复与真实手持漏检，并完成固定完整视频的同条件对照及性能/机位/素材门禁。单次训练约3–4分钟不意味着几分钟就能达标；当前不能给出有依据的上线日期，也不承诺再跑固定轮数必然改善。

下一步先补齐完整验证和困难训练来源，再将生产权重与候选作为固定对照一起评测；后续即使候选从中间版本初始化，也必须将“初始化权重对照”与“生产权重对照”分开命名和报告。

## 证据和运行边界

本轮两个相同对照运行各40次源图推理及40次源NMS，共80次；第二次修正库设置目录创建顺序，并断言设置目录在本地运行根内。第一次因目录父级尚不存在，Ultralytics回退/tmp/Ultralytics，保留全部原始结果；第二次40份原始记录逐项完全相同。该问题未被隐藏为失败训练或质量提升。没有新增训练、改标签、改生产参数、替换模型或付费语义调用。

主证据：`/srv/sentinel-data/VisionCortex3090Ti/Runtime/Project-Detector-Pilot-20260907/diagnostics-goal57-localsettings/`的receipt.json、preflight.json、frozen-inputs.json、call-ledger.json、weight-metadata.json、training-parameter-audit.json、production-comparison-summary.json、independent-count-check.json、settings-repeat-check.json及visual-review-receipt.json。

分支codex/rtx3050-device-delivery-20260904，HEAD 7db0dcd052891e0e189c5e931a27558256fea032，保留未提交工作；总goal active。
