# Awesome LLM for Transportation (Safety & Mobility)

**Official repository for the comprehensive review paper:**

<a href ="https://arxiv.org/abs/2506.06301"> "Large language models and their applications in roadway safety and mobility enhancement: A comprehensive review"</a>
 
> Published in Artificial Intelligence for Transportation (2025)

This repository serves as a **living collection** of the resources, datasets, and papers reviewed in our article. It is curated to help researchers explore the intersection of Large Language Models (LLMs) and Intelligent Transportation Systems (ITS).

## 📚 Citation

If you find this list or our review useful for your research, please cite our paper:

```
@article{KARIM2025100004,
  title = {Large language models and their applications in roadway safety and mobility enhancement: A comprehensive review},
  journal = {Artificial Intelligence for Transportation},
  volume = {1},
  pages = {100004},
  year = {2025},
  issn = {3050-8606},
  doi = {10.1016/j.ait.2025.100004},
  url = {[https://www.sciencedirect.com/science/article/pii/S3050860625000043](https://www.sciencedirect.com/science/article/pii/S3050860625000043)},
  author = {Karim, Muhammad Monjurul and Shi, Yan and Zhang, Shucheng and Wang, Bingzhang and Nasri, Mehrdad and Wang, Yinhai}
}
```

## 📋 Taxonomy & Contents

- [1. Mobility Enhancement](#1-mobility-enhancement)
    
    - [Traffic Flow Prediction](#traffic-flow-prediction)
        
    - [Traffic Data Analysis & Decision Support](#traffic-data-analysis--decision-support)
        
    - [Traffic Signal Control](#traffic-signal-control)
        
    - [Human Mobility & Trip Planning](#human-mobility--trip-planning)
        
    - [Trajectory Prediction](#trajectory-prediction)
        
    - [Simulation & Scenario Generation](#simulation--scenario-generation)
        
    - [Mode Choice & Parking](#mode-choice--parking)
        
- [2. Roadway Safety Enhancement](#2-roadway-safety-enhancement)
    
    - [Crash Data Analysis & Reporting](#crash-data-analysis--reporting)
        
    - [Driver Behavior & Risk Assessment](#driver-behavior--risk-assessment)
        
    - [Pedestrian Safety](#pedestrian-safety)
        
    - [Traffic Rule Compliance](#traffic-rule-compliance)
        
    - [Near-Miss Detection & Scene Understanding](#near-miss-detection--scene-understanding)
        
- [3. Enabling Technologies](#3-enabling-technologies)
    
    - [V2X & Cooperative Driving](#v2x--cooperative-driving)
        
    - [Edge Computing & Efficiency](#edge-computing--efficiency)
        
    - [Explainability & Safety Assurance](#explainability--safety-assurance)
        

## 1. Mobility Enhancement

### Traffic Flow Prediction

_Adapting LLMs for time-series forecasting via textualization, reprogramming, or specialized architectures._

- **TrafficBERT:** Pre-trained model with large-scale data for long-range traffic flow forecasting (Jin et al., 2021) - [ScienceDirect](https://doi.org/10.1016/j.eswa.2021.115738 "null")
    
- **ST-LLM:** Spatial-Temporal Large Language Model for Traffic Prediction (Liu et al., 2024b) - [GitHub](https://github.com/ChenxiLiu-HNU/ST-LLM "null")
    
- **TrafficGPT:** Viewing, processing and interacting with traffic foundation models (Zhang et al., 2024e) - [ScienceDirect](https://www.google.com/search?q=https://doi.org/10.1016/j.tranpol.2024.05.016 "null")
    
- **UrbanGPT:** Spatio-Temporal Large Language Models (Li et al., 2024e) - [arXiv](https://arxiv.org/abs/2403.00813 "null")
    
- **TIME-LLM:** Time Series Forecasting by Reprogramming Large Language Models (Jin et al., 2023) - [arXiv](https://arxiv.org/abs/2310.01728 "null")
    
- **TPLLM:** A Traffic Prediction Framework Based on Pretrained Large Language Models (Ren et al., 2024) - [arXiv](https://arxiv.org/abs/2403.02221 "null")
    
- **Lag-Llama:** Towards foundation models for time series forecasting (Rasul et al., 2023) - [arXiv](https://arxiv.org/abs/2310.08276 "null")
    
- **LLM4TS:** Two-stage fine-tuning for time-series forecasting (Chang et al., 2023) - [arXiv](https://arxiv.org/abs/2308.08469 "null")
    
- **Enhancing traffic prediction with textual data:** (Huang, 2024) - [arXiv](https://arxiv.org/pdf/2405.06719?"null")
    

### Traffic Data Analysis & Decision Support

_Natural language interfaces for querying databases and orchestrating analysis tools._

- **Mobility ChatBot:** Supporting decision making in mobility data with chatbots (Padoan et al., 2024) - [MotionAnalytica](https://www.motionanalytica.com/mobility-chatbot-supporting-decision-making-in-mobility-data-with-chatbots/ "null")
    
- **TP-GPT:** Real-time data informed intelligent chatbot for transportation surveillance (Wang et al., 2024a) - [arXiv](https://arxiv.org/abs/2405.03076 "null")
    
- **IDM-GPT:** Independent Mobility GPT for customized traffic mobility analysis (Yang et al., 2025a) - [arXiv](https://arxiv.org/abs/2502.18652 "null")
    
- **Text-to-SQL for ITS:** (Cai et al., 2022; Li et al., 2023a) - [Google Scholar](https://www.google.com/search?q=https://scholar.google.com/scholar%3Fq%3DSTAR%2BSQL%2Bguided%2Bpre-training "null")
    
- **ChatTraffic:** Text-to-traffic generation via diffusion model (Zhang et al., 2024a) - [GitHub](https://github.com/ChyaZhang/ChatTraffic "null")
    
- **Transit Customer Feedback (MetRoBERTa):** (Leong et al., 2024) - [Google Scholar](https://www.google.com/search?q=https://scholar.google.com/scholar%3Fq%3DMetRoBERTa%2BLeong%2B2024 "null")
    
- **Transit Service Enhancement:** (Wang & Shalaby, 2024) - [Google Scholar](https://www.google.com/search?q=https://scholar.google.com/scholar%3Fq%3DLeveraging%2Blarge%2Blanguage%2Bmodels%2Bfor%2Benhancing%2Bpublic%2Btransit%2Bservices%2BWang%2B2024 "null")
    

### Traffic Signal Control

_LLMs acting as decision-making agents or assisting in control logic design._

- **LLMLight:** Large Language Models as Traffic Signal Control Agents (Lai et al., 2023) - [arXiv](https://arxiv.org/abs/2305.07436 "null")
    
- **Digital Traffic Engineers:** Large Language Model-Powered Digital Traffic Engineers (Dai et al., 2024) - [IEEE](https://ieeexplore.ieee.org/document/10472483 "null")
    
- **LA-Light:** Leveraging LLM Capabilities for Human-Mimetic Traffic Signal Control (Wang et al., 2024d) - [arXiv](https://arxiv.org/abs/2403.08337 "null")
    
- **iLLM-TSC:** Integration Reinforcement Learning and LLM for Traffic Signal Control (Pang et al., 2024) - [arXiv](https://arxiv.org/abs/2407.06025 "null")
    
- **RAGTraffic:** Utilizing RAG for Intelligent Traffic Signal Control (Zhang et al., 2024f) - [Google Scholar](https://www.google.com/search?q=https://scholar.google.com/scholar%3Fq%3DRAGTraffic%2BUtilizing%2BRetrieval-Augmented%2BGeneration%2Bfor%2BIntelligent%2BTraffic%2BSignal%2BControl "null")
    
- **LLM-Assisted Arterial Control:** (Tang et al., 2024a) - [Google Scholar](https://www.google.com/search?q=https://scholar.google.com/scholar%3Fq%3DLarge%2Blanguage%2Bmodel-assisted%2Barterial%2Btraffic%2Bsignal%2Bcontrol%2BTang%2B2024 "null")
    
- **PromptGAT:** LLM powered sim-to-real transfer for traffic signal control (Da et al., 2023) - [NSF PAR](https://par.nsf.gov/servlets/purl/10525336 "null")
    
- **4D Traffic Control:** (Masri et al., 2025) - [Google Scholar](https://www.google.com/search?q=https://scholar.google.com/scholar%3Fq%3DMasri%2B2025%2BTraffic%2BControl%2BLLM "null")
    

### Human Mobility & Trip Planning

_Predicting next locations, synthesizing travel patterns, and itinerary planning._

- **NextLocLLM:** Next Location Prediction using LLMs (Liu et al., 2024c) - [arXiv](https://arxiv.org/abs/2410.09129 "null")
    
- **AgentMove:** Predicting Human Mobility Anywhere (Feng et al., 2024) - [arXiv](https://arxiv.org/abs/2408.13986 "null")
    
- **HMP-LLM:** Human Mobility Prediction based on Pre-trained LLMs (Zhong et al., 2024) - [IEEE](https://ieeexplore.ieee.org/document/10620835 "null")
    
- **Mobility-LLM:** Learning visiting intentions and travel preference (Gong et al., 2024) - [Google Scholar](https://www.google.com/search?q=https://scholar.google.com/scholar%3Fq%3DMobility-LLM%2BGong%2B2024 "null")
    
- **ITINERA:** Synergizing spatial optimization with LLMs for itinerary planning (Tang et al., 2024c) - [Google Scholar](https://www.google.com/search?q=https://scholar.google.com/scholar%3Fq%3DITINERA%2BTang%2B2024 "null")
    
- **Intelligent Trip Planning:** (Pio et al., 2024) - [Google Scholar](https://www.google.com/search?q=https://scholar.google.com/scholar%3Fq%3DPio%2B2024%2BIntelligent%2BTrip%2BPlanning "null")
    
- **LingoTrip:** Spatiotemporal context prompt driven LLM for trip prediction (Qin et al., 2025) - [Google Scholar](https://www.google.com/search?q=https://scholar.google.com/scholar%3Fq%3DLingoTrip%2BQin%2B2025 "null")
    

### Trajectory Prediction

_Predicting movements of vehicles and pedestrians using sequence modeling._

- **Traj-LLM:** Empowering Trajectory Prediction with Pre-trained LLMs (Lan et al., 2024) - [ICLR 2025](https://proceedings.iclr.cc/paper_files/paper/2025/file/6bbefb73c0ede70635823a18426b9208-Paper-Conference.pdf "null")
    
- **LC-LLM:** Explainable Lane-Change Intention and Trajectory Predictions (Peng et al., 2025) - [arXiv](https://arxiv.org/abs/2403.18344 "null")
    
- **HighwayLLM:** Decision-Making and Navigation in Highway Driving (Yildirim et al., 2024) - [arXiv](https://arxiv.org/abs/2405.13547 "null")
    
- **LG-Traj:** LLM Guided Pedestrian Trajectory Prediction (Chib & Singh, 2024) - [CVF](https://openaccess.thecvf.com/content/ICCV2025W/CDEL/papers/Chib_LG-Traj_LLM_Guided_Pedestrian_Trajectory_Prediction_ICCVW_2025_paper.pdf "null")
    
- **Trajectory-LLM:** A language-based data generator for trajectory prediction (Yang et al., 2025b) - [ICLR 2025](https://proceedings.iclr.cc/paper_files/paper/2025/hash/6bbefb73c0ede70635823a18426b9208-Abstract-Conference.html "null")
    
- **InteractTraj:** Language-driven interactive traffic trajectory generation (Xia et al., 2024) - [Google Scholar](https://www.google.com/search?q=https://scholar.google.com/scholar%3Fq%3DInteractTraj%2BXia%2B2024 "null")
    

### Simulation & Scenario Generation

_Using natural language to generate scenarios for simulation (e.g., SUMO, CARLA)._

- **ChatSUMO:** LLM for Automating Traffic Scenario Generation (Li et al., 2024b) - [IEEE](https://ieeexplore.ieee.org/document/10588471 "null")
    
- **Text2Scenario:** Text-Driven Scenario Generation for AD Test (Cai et al., 2025) - [Google Scholar](https://www.google.com/search?q=https://scholar.google.com/scholar%3Fq%3DText2Scenario%2BCai%2B2025 "null")
    
- **LLMScenario:** LLM Driven Scenario Generation (Chang et al., 2024) - [Google Scholar](https://www.google.com/search?q=https://scholar.google.com/scholar%3Fq%3DLLMScenario%2BChang%2B2024 "null")
    
- **ProSim:** Promptable Closed-Loop Traffic Simulation (Tan et al.) - [Google Scholar](https://www.google.com/search?q=https://scholar.google.com/scholar%3Fq%3DProSim%2BTan%2BTraffic%2BSimulation "null")
    
- **Traffic Scene Generation from NL:** (Ruan et al., 2024) - [Google Scholar](https://www.google.com/search?q=https://scholar.google.com/scholar%3Fq%3DRuan%2B2024%2BTraffic%2BScene%2BGeneration "null")
    
- **SeGPT:** Scenario Engineer with ChatGPT (Li et al., 2024d) - [Google Scholar](https://www.google.com/search?q=https://scholar.google.com/scholar%3Fq%3DSeGPT%2BLi%2B2024 "null")
    
- **DIAVIO:** LLM-empowered diagnosis of safety violations (Lu et al., 2024) - [Google Scholar](https://www.google.com/search?q=https://scholar.google.com/scholar%3Fq%3DDIAVIO%2BLu%2B2024 "null")
    

### Mode Choice & Parking

- **Travel Mode Choice with LLMs:** (Mo et al., 2023; Liu et al., 2024d; Zhai et al., 2024) - [Google Scholar](https://www.google.com/search?q=https://scholar.google.com/scholar%3Fq%3DMo%2B2023%2BTravel%2BMode%2BChoice%2BLLM "null")
    
- **DelayPTC-LLM:** Metro passenger travel choice under delays (Chen et al., 2024) - [arXiv](https://arxiv.org/abs/2410.00052 "null")
    
- **Parking Planning Agent:** (Jin & Ma, 2024) - [Google Scholar](https://www.google.com/search?q=https://scholar.google.com/scholar%3Fq%3DJin%2BMa%2B2024%2BParking%2BPlanning "null")
    
- **Simulating Parking Search:** (Fulman et al., n.d.) - [Google Scholar](https://www.google.com/search?q=https://scholar.google.com/scholar%3Fq%3DFulman%2BParking%2BSearch%2BLLM "null")
    

## 2. Roadway Safety Enhancement

### Crash Data Analysis & Reporting

_Extracting structured insights from unstructured narratives and reports._

- **CrashLLM:** Learning Traffic Crashes as Language (Fan et al., 2024) - [arXiv](https://arxiv.org/abs/2406.10789 "null")
    
- **TrafficSafetyGPT:** Tuning a Pre-trained LLM to a Domain-Specific Expert (Zheng et al., 2023) - [arXiv](https://arxiv.org/abs/2307.15311 "null")
    
- **Crash Severity Analysis with CoT:** (Zhen et al., 2024) - [Google Scholar](https://www.google.com/search?q=https://scholar.google.com/scholar%3Fq%3DZhen%2B2024%2BCrash%2BSeverity%2BAnalysis "null")
    
- **AccidentGPT:** Accident Analysis from V2X Perception (Wang et al., 2023) - [arXiv](https://arxiv.org/abs/2312.13156 "null")
    
- **Uncovering Underreporting in Crashes:** (Arteaga & Park, 2025) - [ScienceDirect](https://www.google.com/search?q=https://doi.org/10.1016/j.jsr.2024.12.006 "null")
    
- **AutoRepo:** Automated Construction Reporting (Pu et al., 2024) - [Google Scholar](https://www.google.com/search?q=https://scholar.google.com/scholar%3Fq%3DAutoRepo%2BPu%2B2024 "null")
    
- **RTC Analysis using Twitter:** (Jaradat et al., 2024b) - [Google Scholar](https://www.google.com/search?q=https://scholar.google.com/scholar%3Fq%3DJaradat%2B2024%2BTwitter%2BCrash%2BAnalysis "null")
    

### Driver Behavior & Risk Assessment

_Monitoring driver states (distraction, fatigue) and risky behaviors._

- **DDLM:** Integrating Visual LLM for Driver Behavior Analysis (Zhang et al., 2024b) - [ScienceDirect](https://www.google.com/search?q=https://doi.org/10.1016/j.aap.2024.107497 "null")
    
- **SurrealDriver:** Generative Driver Agent Framework (Jin et al., 2024) - [arXiv](https://arxiv.org/abs/2309.13193 "null")
    
- **LanguageMPC:** LLMs as Decision Makers for Autonomous Driving (Sha et al., 2023) - [arXiv](https://arxiv.org/abs/2310.03026 "null")
    
- **DriveVLM:** Multi-Frame VLM for Driver Behavior Analysis (Takato et al., 2024) - [arXiv](https://arxiv.org/abs/2402.12289 "null")
    
- **Driving Style Alignment:** (Yang et al., 2024b) - [Google Scholar](https://www.google.com/search?q=https://scholar.google.com/scholar%3Fq%3DYang%2B2024%2BDriving%2BStyle%2BAlignment "null")
    
- **Text-to-Drive:** (Nguyen et al., 2024) - [Google Scholar](https://www.google.com/search?q=https://scholar.google.com/scholar%3Fq%3DText-to-Drive%2BNguyen%2B2024 "null")
    
- **SenseRAG:** Environmental Knowledge Bases for AD (Luo et al., 2025) - [CVF](https://openaccess.thecvf.com/content/WACV2025W/LLVMAD/html/Luo_SenseRAG_Constructing_Environmental_Knowledge_Bases_with_Proactive_Querying_for_LLM-Based_WACVW_2025_paper.html "null")
    

### Pedestrian Safety

_Pedestrian intention prediction and safety monitoring._

- **PedVLM:** Pedestrian VLM for Intentions Prediction (Munir et al., 2025) - [OpenReview](https://openreview.net/pdf/68989e2847e3bf02f5fb99a70f2a430574b5f374.pdf "null")
    
- **VTPM:** Video-to-Text Pedestrian Monitoring (Abdelrahman et al., 2024) - [arXiv](https://arxiv.org/abs/2408.11649 "null")
    
- **RAG-Based VRU Prediction:** (Hussien et al., 2025) - [Google Scholar](https://www.google.com/search?q=https://scholar.google.com/scholar%3Fq%3DHussien%2B2025%2BRAG%2BPedestrian "null")
    
- **Walk-the-Talk:** LLM Driven Pedestrian Motion Generation (Ramesh & Flohr, 2024) - [IEEE](https://ieeexplore.ieee.org/document/10588625 "null")
    
- **Safety-aware street crossing (Blind/Low-Vision):** (Hwang et al., 2024) - [Google Scholar](https://www.google.com/search?q=https://scholar.google.com/scholar%3Fq%3DHwang%2B2024%2BSafety%2Baware%2Bstreet%2Bcrossing "null")
    

### Traffic Rule Compliance

_Translating natural language traffic laws into formal logic for AVs._

- **TR2MTL:** Metric Temporal Logic Formalization of Traffic Rules (Manas et al., 2024) - [IEEE](https://ieeexplore.ieee.org/document/10588722 "null")
    
- **Driving with Regulation:** Interpretable Decision-Making (Cai et al., 2024) - [arXiv](https://arxiv.org/abs/2410.04759 "null")
    
- **LLM-Enhanced RL for Ramp Merging:** (Yin et al., 2024) - [Google Scholar](https://www.google.com/search?q=https://scholar.google.com/scholar%3Fq%3DYin%2B2024%2BLLM%2BReinforcement%2BLearning%2BTraffic "null")
    

### Near-Miss Detection & Scene Understanding

- **Near-Miss Detection with MLLMs:** (Jaradat et al., 2025) - [ResearchGate](https://www.researchgate.net/publication/387718170_Leveraging_Deep_Learning_and_Multimodal_Large_Language_Models_for_Near-Miss_Detection_Using_Crowdsourced_Videos "null")
    
- **Automated Detection of Safety-Critical Events:** (Abu Tami et al., 2024) - [arXiv](https://arxiv.org/abs/2406.13894 "null")
    
- **MAPLM:** Real-world large-scale vision-language benchmark (Cao et al., 2024) - [CVF](https://openaccess.thecvf.com/content/CVPR2024/html/Cao_MAPLM_A_Real-World_Large-Scale_Vision-Language_Benchmark_for_Map_and_Traffic_CVPR_2024_paper.html "null")
    
- **LingoQA:** Visual Question Answering for Autonomous Driving (Marcu et al., 2024) - [Google Scholar](https://www.google.com/search?q=https://scholar.google.com/scholar%3Fq%3DLingoQA%2BMarcu%2B2024 "null")
    
- **Traffic-Domain VQA:** (Qasemi et al., 2023; Guo et al., 2024a) - [Google Scholar](https://www.google.com/search?q=https://scholar.google.com/scholar%3Fq%3DTraffic-domain%2Bvideo%2Bquestion%2Banswering "null")
    

## 3. Enabling Technologies

### V2X & Cooperative Driving

- **V2X-LLM:** Enhancing V2X Integration and Understanding (Wu et al., 2025) - [arXiv](https://arxiv.org/abs/2503.02239 "null")
    
- **CoDrivingLLM:** Interactive and Learnable Cooperative Driving (Fang et al., 2025) - [Google Scholar](https://www.google.com/search?q=https://scholar.google.com/scholar%3Fq%3DCoDrivingLLM%2BFang%2B2025 "null")
    
- **V2X-VLM:** End-to-end V2X cooperative driving (You et al., 2024) - [Google Scholar](https://www.google.com/search?q=https://scholar.google.com/scholar%3Fq%3DV2X-VLM%2BYou%2B2024 "null")
    
- **BlockLLM:** Decentralized vehicular network architecture (Arshad & Halim, 2025) - [Google Scholar](https://www.google.com/search?q=https://scholar.google.com/scholar%3Fq%3DBlockLLM%2BArshad%2B2025 "null")
    

### Edge Computing & Efficiency

- **Edge Computing Enabled Traffic Prediction:** (Rong et al., 2024a) - [IEEE](https://www.google.com/search?q=https://ieeexplore.ieee.org/document/10636272 "null")
    
- **Lightweight Spatio-temporal LLM on Edge:** (Rong et al., 2024b) - [Google Scholar](https://www.google.com/search?q=https://scholar.google.com/scholar%3Fq%3DLSGLLM-E%2BRong%2B2024 "null")
    
- **Efficient Driving Behavior Narration on Edge:** (Huang et al., 2024a) - [Google Scholar](https://www.google.com/search?q=https://scholar.google.com/scholar%3Fq%3DHuang%2B2024%2BEdge%2BLLM%2BDriving "null")
    
- **ScaleLLM:** Resource-frugal LLM serving (Yao et al., 2024) - [Google Scholar](https://www.google.com/search?q=https://scholar.google.com/scholar%3Fq%3DScaleLLM%2BYao%2B2024 "null")
    

### Explainability & Safety Assurance

- **Towards Explainable Traffic Flow Prediction:** (Guo et al., 2024b) - [Google Scholar](https://www.google.com/search?q=https://scholar.google.com/scholar%3Fq%3DGuo%2B2024%2BExplainable%2BTraffic%2BFlow%2BPrediction "null")
    
- **Safety Case Generation with GPT-4:** (Sivakumar et al., 2024) - [Google Scholar](https://www.google.com/search?q=https://scholar.google.com/scholar%3Fq%3DSivakumar%2B2024%2BSafety%2BCase%2BGeneration "null")
    
- **CrashEventLLM:** Predicting system crashes (Mudgal et al., 2024) - [arXiv](https://arxiv.org/abs/2407.15716 "null")
    

## 🤝 Contributing

This is a living repository. We welcome contributions from the community!

1. **Fork** this repository.
    
2. **Add** your paper to the relevant category.
    
3. Submit a **Pull Request**.
    

