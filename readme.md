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

- **TrafficBERT:** Pre-trained model with large-scale data for long-range traffic flow forecasting (Jin et al., 2021) - [ScienceDirect](https://doi.org/10.1016/j.eswa.2021.115738)
- **ST-LLM:** Spatial-Temporal Large Language Model for Traffic Prediction (Liu et al., 2024b) - [GitHub](https://github.com/ChenxiLiu-HNU/ST-LLM)
- **TrafficGPT:** Viewing, processing and interacting with traffic foundation models (Zhang et al., 2024e) - [ScienceDirect](https://www.sciencedirect.com/science/article/pii/S0967070X24000726)
- **UrbanGPT:** Spatio-Temporal Large Language Models (Li et al., 2024e) - [arXiv](https://arxiv.org/abs/2403.00813)
- **TIME-LLM:** Time Series Forecasting by Reprogramming Large Language Models (Jin et al., 2023) - [arXiv](https://arxiv.org/abs/2310.01728)
- **TPLLM:** A Traffic Prediction Framework Based on Pretrained Large Language Models (Ren et al., 2024) - [arXiv](https://arxiv.org/abs/2403.02221)
- **Lag-Llama:** Towards foundation models for time series forecasting (Rasul et al., 2023) - [arXiv](https://arxiv.org/abs/2310.08276)
- **LLM4TS:** Two-stage fine-tuning for time-series forecasting (Chang et al., 2023) - [arXiv](https://arxiv.org/abs/2308.08469)
- **Enhancing traffic prediction with textual data:** (Huang, 2024) - [arXiv](https://arxiv.org/abs/2405.06719)

### Traffic Data Analysis & Decision Support

_Natural language interfaces for querying databases and orchestrating analysis tools._

- **Mobility ChatBot:** Supporting decision making in mobility data with chatbots (Padoan et al., 2024) - [IEEE](https://ieeexplore.ieee.org/document/10591685)
- **TP-GPT:** Real-time data informed intelligent chatbot for transportation surveillance (Wang et al., 2024a) - [arXiv](https://arxiv.org/abs/2405.03076)
- **IDM-GPT:** Independent Mobility GPT for customized traffic mobility analysis (Yang et al., 2025a) - [arXiv](https://arxiv.org/abs/2502.18652)
- **STAR:** Text-to-SQL for ITS (Cai et al., 2022) - [arXiv](https://arxiv.org/abs/2210.11888)
- **BIRD:** Text-to-SQL benchmark (Li et al., 2023a) - [arXiv](https://arxiv.org/abs/2305.03111)
- **ChatTraffic:** Text-to-traffic generation via diffusion model (Zhang et al., 2024a) - [GitHub](https://github.com/ChyaZhang/ChatTraffic)
- **MetRoBERTa:** Transit Customer Feedback Analysis (Leong et al., 2024) - [Paper](https://journals.sagepub.com/doi/10.1177/03611981241234622)
- **Leveraging LLMs for Transit Services:** (Wang & Shalaby, 2024) - [arXiv](https://arxiv.org/abs/2410.14147)

### Traffic Signal Control

_LLMs acting as decision-making agents or assisting in control logic design._

- **LLMLight:** Large Language Models as Traffic Signal Control Agents (Lai et al., 2023) - [arXiv](https://arxiv.org/abs/2305.07436)
- **Digital Traffic Engineers:** Large Language Model-Powered Digital Traffic Engineers (Dai et al., 2024) - [IEEE](https://ieeexplore.ieee.org/document/10660467)
- **LA-Light:** Leveraging LLM Capabilities for Human-Mimetic Traffic Signal Control (Wang et al., 2024d) - [arXiv](https://arxiv.org/abs/2403.08337)
- **iLLM-TSC:** Integration Reinforcement Learning and LLM for Traffic Signal Control (Pang et al., 2024) - [arXiv](https://arxiv.org/abs/2407.06025)
- **RAGTraffic:** Utilizing RAG for Intelligent Traffic Signal Control (Zhang et al., 2024f) - [IEEE](https://ieeexplore.ieee.org/document/10919289)
- **LLM-Assisted Arterial Control:** (Tang et al., 2024a) - [IEEE](https://ieeexplore.ieee.org/document/10488379)
- **PromptGAT:** LLM powered sim-to-real transfer for traffic signal control (Da et al., 2023) - [arXiv](https://arxiv.org/abs/2308.14284)
- **4D Traffic Control Framework:** (Masri et al., 2025) - [Paper](https://www.mdpi.com/2624-8921/7/1/11)

### Human Mobility & Trip Planning

_Predicting next locations, synthesizing travel patterns, and itinerary planning._

- **NextLocLLM:** Next Location Prediction using LLMs (Liu et al., 2024c) - [arXiv](https://arxiv.org/abs/2410.09129)
- **AgentMove:** Predicting Human Mobility Anywhere (Feng et al., 2024) - [arXiv](https://arxiv.org/abs/2408.13986)
- **HMP-LLM:** Human Mobility Prediction based on Pre-trained LLMs (Zhong et al., 2024) - [IEEE](https://ieeexplore.ieee.org/document/10778764)
- **Mobility-LLM:** Learning visiting intentions and travel preference (Gong et al., 2024) - [NeurIPS](https://proceedings.neurips.cc/paper_files/paper/2024/file/3fb6c52aeb11e09053c16eabee74dd7b-Paper-Conference.pdf)
- **ITINERA:** Synergizing spatial optimization with LLMs for itinerary planning (Tang et al., 2024c) - [arXiv](https://arxiv.org/abs/2402.07204)
- **Intelligent Trip Planning:** Framework for LLM-powered trip planning (Pio et al., 2024) - [IEEE](https://ieeexplore.ieee.org/document/10771388)
- **LingoTrip:** Spatiotemporal context prompt driven LLM for trip prediction (Qin et al., 2025) - [Paper](https://www.sciencedirect.com/science/article/pii/S1077291X24001309)

### Trajectory Prediction

_Predicting movements of vehicles and pedestrians using sequence modeling._

- **Traj-LLM:** Empowering Trajectory Prediction with Pre-trained LLMs (Lan et al., 2024) - [arXiv](https://arxiv.org/abs/2405.04909)
- **LC-LLM:** Explainable Lane-Change Intention and Trajectory Predictions (Peng et al., 2025) - [arXiv](https://arxiv.org/abs/2403.18344)
- **HighwayLLM:** Decision-Making and Navigation in Highway Driving (Yildirim et al., 2024) - [arXiv](https://arxiv.org/abs/2405.13547)
- **LG-Traj:** LLM Guided Pedestrian Trajectory Prediction (Chib & Singh, 2024) - [arXiv](https://arxiv.org/abs/2403.08032)
- **Trajectory-LLM:** A language-based data generator for trajectory prediction (Yang et al., 2025b) - [OpenReview](https://openreview.net/forum?id=vU1oYVIHi3)
- **InteractTraj:** Language-driven interactive traffic trajectory generation (Xia et al., 2024) - [NeurIPS](https://proceedings.neurips.cc/paper_files/paper/2024)

### Simulation & Scenario Generation

_Using natural language to generate scenarios for simulation (e.g., SUMO, CARLA)._

- **ChatSUMO:** LLM for Automating Traffic Scenario Generation (Li et al., 2024b) - [IEEE](https://ieeexplore.ieee.org/document/10588471)
- **Text2Scenario:** Text-Driven Scenario Generation for AD Test (Cai et al., 2025) - [arXiv](https://arxiv.org/abs/2503.02911)
- **LLMScenario:** LLM Driven Scenario Generation (Chang et al., 2024) - [IEEE](https://ieeexplore.ieee.org/document/10443396)
- **ProSim:** Promptable Closed-Loop Traffic Simulation (Tan et al., 2024) - [CoRL](https://openreview.net/forum?id=CMmiMDj4Yx)
- **Traffic Scene Generation from NL:** (Ruan et al., 2024) - [arXiv](https://arxiv.org/abs/2409.09575)
- **SeGPT:** Scenario Engineer with ChatGPT (Li et al., 2024d) - [IEEE](https://ieeexplore.ieee.org/document/10422291)
- **DIAVIO:** LLM-empowered diagnosis of safety violations (Lu et al., 2024) - [ACM](https://dl.acm.org/doi/10.1145/3650212.3680323)

### Mode Choice & Parking

- **Travel Mode Choice with LLMs:** (Mo et al., 2023) - [arXiv](https://arxiv.org/abs/2312.00819)
- **Can LLMs Capture Human Travel Behavior:** (Liu et al., 2024d) - [SSRN](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=4937575)
- **Enhancing Travel Choice with LLMs:** (Zhai et al., 2024) - [arXiv](https://arxiv.org/abs/2406.13558)
- **DelayPTC-LLM:** Metro passenger travel choice under delays (Chen et al., 2024) - [arXiv](https://arxiv.org/abs/2410.00052)
- **Parking Planning Agent:** (Jin & Ma, 2024) - [Paper](https://www.sciencedirect.com/science/article/pii/S2210670724006255)
- **Simulating Parking Search:** (Fulman et al., 2024) - [SSRN](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=5018154)

## 2. Roadway Safety Enhancement

### Crash Data Analysis & Reporting

_Extracting structured insights from unstructured narratives and reports._

- **CrashLLM:** Learning Traffic Crashes as Language (Fan et al., 2024) - [arXiv](https://arxiv.org/abs/2406.10789)
- **TrafficSafetyGPT:** Tuning a Pre-trained LLM to a Domain-Specific Expert (Zheng et al., 2023) - [arXiv](https://arxiv.org/abs/2307.15311)
- **Crash Severity Analysis with CoT:** (Zhen et al., 2024) - [Paper](https://www.mdpi.com/2073-431X/13/9/232)
- **AccidentGPT:** Accident Analysis from V2X Perception (Wang et al., 2023) - [arXiv](https://arxiv.org/abs/2312.13156)
- **Uncovering Underreporting in Crashes:** (Arteaga & Park, 2025) - [Paper](https://doi.org/10.1016/j.jsr.2024.12.006)
- **Hybrid ML for Crash Severity:** (Grigorev et al., 2024) - [arXiv](https://arxiv.org/abs/2403.12536)
- **RTC Analysis using Twitter:** (Jaradat et al., 2024b) - [Paper](https://www.mdpi.com/2624-6511/7/5/2465)

### Driver Behavior & Risk Assessment

_Monitoring driver states (distraction, fatigue) and risky behaviors._

- **DDLM:** Integrating Visual LLM for Driver Behavior Analysis (Zhang et al., 2024b) - [Paper](https://doi.org/10.1016/j.aap.2024.107497)
- **SurrealDriver:** Generative Driver Agent Framework (Jin et al., 2024) - [IEEE](https://ieeexplore.ieee.org/document/10802159)
- **LanguageMPC:** LLMs as Decision Makers for Autonomous Driving (Sha et al., 2023) - [arXiv](https://arxiv.org/abs/2310.03026)
- **Multi-Frame VLM for Driver Analysis:** (Takato et al., 2024) - [arXiv](https://arxiv.org/abs/2408.01682)
- **Driving Style Alignment:** (Yang et al., 2024b) - [IEEE](https://ieeexplore.ieee.org/document/10802238)
- **Text-to-Drive:** (Nguyen et al., 2024) - [IEEE](https://ieeexplore.ieee.org/document/10802328)
- **SenseRAG:** Environmental Knowledge Bases for AD (Luo et al., 2025) - [CVF](https://openaccess.thecvf.com/content/WACV2025W/LLVMAD/html/Luo_SenseRAG_Constructing_Environmental_Knowledge_Bases_with_Proactive_Querying_for_LLM-Based_WACVW_2025_paper.html)

### Pedestrian Safety

_Pedestrian intention prediction and safety monitoring._

- **PedVLM:** Pedestrian VLM for Intentions Prediction (Munir et al., 2025) - [IEEE](https://ieeexplore.ieee.org/document/10839577)
- **VTPM:** Video-to-Text Pedestrian Monitoring (Abdelrahman et al., 2024) - [arXiv](https://arxiv.org/abs/2408.11649)
- **RAG-Based VRU Prediction:** (Hussien et al., 2025) - [Paper](https://doi.org/10.1016/j.eswa.2024.125914)
- **Walk-the-Talk:** LLM Driven Pedestrian Motion Generation (Ramesh & Flohr, 2024) - [IEEE](https://ieeexplore.ieee.org/document/10588625)
- **GPT-4V for Street Crossing Safety:** (Hwang et al., 2024) - [IEEE](https://ieeexplore.ieee.org/document/10611726)

### Traffic Rule Compliance

_Translating natural language traffic laws into formal logic for AVs._

- **TR2MTL:** Metric Temporal Logic Formalization of Traffic Rules (Manas et al., 2024) - [IEEE](https://ieeexplore.ieee.org/document/10588722)
- **Driving with Regulation:** Interpretable Decision-Making (Cai et al., 2024) - [arXiv](https://arxiv.org/abs/2410.04759)
- **LLM-Enhanced RL for Ramp Merging:** (Yin et al., 2024) - [Springer](https://link.springer.com/chapter/10.1007/978-981-96-0351-0_27)

### Near-Miss Detection & Scene Understanding

- **Near-Miss Detection with MLLMs:** (Jaradat et al., 2025) - [IEEE](https://ieeexplore.ieee.org/document/10839629)
- **Automated Detection of Safety-Critical Events:** (Abu Tami et al., 2024) - [Paper](https://www.mdpi.com/2624-8921/6/3/1571)
- **MAPLM:** Real-world large-scale vision-language benchmark (Cao et al., 2024) - [CVF](https://openaccess.thecvf.com/content/CVPR2024/html/Cao_MAPLM_A_Real-World_Large-Scale_Vision-Language_Benchmark_for_Map_and_Traffic_CVPR_2024_paper.html)
- **LingoQA:** Visual Question Answering for Autonomous Driving (Marcu et al., 2024) - [ECCV](https://link.springer.com/chapter/10.1007/978-3-031-72764-1_15)
- **TRIVIA:** Traffic-Domain VQA with Knowledge Injection (Qasemi et al., 2023) - [arXiv](https://arxiv.org/abs/2307.09636)
- **Traffic Event VQA:** (Guo et al., 2024a) - [IEEE](https://ieeexplore.ieee.org/document/10543362)

## 3. Enabling Technologies

### V2X & Cooperative Driving

- **V2X-LLM:** Enhancing V2X Integration and Understanding (Wu et al., 2025) - [arXiv](https://arxiv.org/abs/2503.02239)
- **CoDrivingLLM:** Interactive and Learnable Cooperative Driving (Fang et al., 2025) - [IEEE](https://ieeexplore.ieee.org/document/10817854)
- **V2X-VLM:** End-to-end V2X cooperative driving (You et al., 2024) - [arXiv](https://arxiv.org/abs/2408.09251)
- **BlockLLM:** Decentralized vehicular network architecture (Arshad & Halim, 2025) - [Paper](https://doi.org/10.1016/j.compeleceng.2024.110027)

### Edge Computing & Efficiency

- **Edge Computing Enabled Traffic Prediction:** (Rong et al., 2024a) - [IEEE](https://ieeexplore.ieee.org/document/10636272)
- **Lightweight Spatio-temporal LLM on Edge:** (Rong et al., 2024b) - [IEEE](https://ieeexplore.ieee.org/document/10830685)
- **Efficient Driving Behavior Narration on Edge:** (Huang et al., 2024a) - [arXiv](https://arxiv.org/abs/2409.20364)
- **ScaleLLM:** Resource-frugal LLM serving (Yao et al., 2024) - [arXiv](https://arxiv.org/abs/2408.00008)

### Explainability & Safety Assurance

- **xTP-LLM:** Explainable Traffic Flow Prediction (Guo et al., 2024b) - [Paper](https://doi.org/10.1016/j.ctrf.2024.100150)
- **Safety Case Generation with GPT-4:** (Sivakumar et al., 2024) - [Paper](https://doi.org/10.1016/j.eswa.2024.124653)
- **CrashEventLLM:** Predicting system crashes (Mudgal et al., 2024) - [IEEE](https://ieeexplore.ieee.org/document/10768645)

## 🤝 Contributing

This is a living repository. We welcome contributions from the community!

1. **Fork** this repository.
2. **Add** your paper to the relevant category.
3. Submit a **Pull Request**.

## 📧 Contact

For questions or suggestions, please open an issue or contact the authors.

## 📜 License

This repository is maintained for academic purposes. Please cite our review paper if you use this resource in your research.

        
