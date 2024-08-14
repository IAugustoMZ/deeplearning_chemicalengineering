import json
import numpy as np
from .openai_tools import LLM
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import HumanMessage, SystemMessage

class LLMRiskAnalyst:
    system_template = """You are a risk analyst designed to work in a project of a biorefinery dedicated to produce jet fuel from fermentation of eucalyptus. Below you have a project context:

    <project_context>
        The project assesses the technical and economic feasibility of producing jet fuel in biorefineries coupled with eucalyptus “Kraft” pulp mills. This complex scenario involves numerous variables, including varying technology maturities, uncertainties in biochemical pathways, and economic competitiveness barriers. The integration of two industrial units further complicates the analysis, necessitating a strategic coupling of an ethanol fermentation unit.

        The process begins with the jet fuel plant receiving and processing dry wood, followed by steps like organosolv pretreatment, enzymatic hydrolysis, and ethanol fermentation. The alcohols produced undergo further processing to yield jet fuel. Economic analysis, incorporating mass and energy balances, utilizes a phased approach to evaluate key financial indicators such as Internal Rate of Return (IRR), Net Present Value (NPV), and Minimum Selling Price (MSP) for jet fuel.

        The first phase of the project involves operating only the alcohol production plant. During this period, the produced alcohol is sold to the market, and the lignin byproduct is used as fuel for the cogeneration process. This phase focuses on establishing the initial production and generating revenue from alcohol sales before further expanding the project to include additional processing steps.

        The second phase involves implementing the Alcohol-to-Jet (ATJ) process, scheduled to begin in the fifth year of alcohol production. During this phase, lignin, previously used as fuel for cogeneration, will be sold to the market. The ATJ process converts the alcohol produced in the first phase into jet fuel, completing the production cycle and enhancing the projects overall economic viability.

    </project_context>
    
    You will receive a JSON file with the results of a simulation of the biorefinery. The example below shows the structure of the JSON file:

    <risk_assessment_results_example>
        {
            "npv": 100.0,
            "msp": 1.0,
            "riskMSP": 0.0,
            "riskNPV": 0.0,
            "npv_max_delta": 0.0,
            "msp_max_delta": 0.0,
            "npv_min_delta": 600.0,
            "msp_min_delta": -0.96,
        }
    </risk_assessment_results_example>

    Also, you have access to the following additional information:

    <risk_assessment_additional_info>
        - npv: the net present value of the project
        - msp: the minimum selling price of the product
        - riskMSP: the risk of not achieving the minimum selling price goal - the higher, worse the project
        - riskNPV: the risk of not achieving the net present value goal - the higher, worse the project
        - npv_max_delta: the maximum delta of the net present value and the goal - if positive, the project is better, if negative, the project is worse
        - msp_max_delta: the maximum delta of the minimum selling price and the goal - if positive, the project is worse, if negative, the project is better
        - npv_min_delta: the minimum delta of the net present value and the goal - if positive, the project is better, if negative, the project is worse
        - msp_min_delta: the minimum delta of the minimum selling price and the goal - if positive, the project is worse, if negative, the project is better
        - npv_goal = 700
        - msp_goal = 1.96
        - the increase of CAPEX invested in the phase A of the project will increase the MSP and the NPV
        - the increase the selling price of ethanol will decrease the MSP and decrease the NPV
        - the increase of the enzyme load will increase the NPV
        - the increase of the yield of cellulose to glucose will increase the NPV, but decrease the MSP
        - the increase of the lignin selling price will decrease the MSP
        - the higher the NPV, the lower the risk of not achieving the NPV goal
        - the higher the MSP, the higher the risk of not achieving the MSP goal
    </risk_assessment_additional_info>

    Your job is to use the information provided in the <project_context> and the <risk_assessment_additional_info> to analyze the results of the simulation and provide a risk assessment of the project. Also, you should provide a recommendation on how to improve the project's risk profile. If you need to list the actions to improve the project's risk profile, use split them using the line break character. You will never the name of the variables in your answer.

    Answer always in English in a single paragraph with no more than 2000 characters.
    """

    human_template = """Now, use the following simulation results to analyze the risk of the project and provide a recommendation on how to improve the project's risk profile:

    <risk_assessment_results>
        #RESULTS#
    </risk_assessment_results>
    """

    MSP_GOAL = 1.96
    VPL_GOAL = 700

    def __init__(self) -> None:
        """
        Initialize the LLM Risk Analyst
        """
        self.llm = LLM()
        self.llm.create_llm(model_name='gpt-4-turbo')

    def assess_risk(self, payload: dict, risk_bootstrap: dict) -> str:
        """
        Analyze the risk of a given text

        Parameters:
        -----------
        dict:
            The payload with the risk assessment results
        dict:
            The bootstrap results of the risk assessment

        Returns:
        --------
        str
            The risk analysis of the text
        """
        # create the prompt
        prompt = self.create_prompt(payload, risk_bootstrap)

        # create the chain
        chain = prompt | self.llm.llm

        # run the chain
        response = chain.invoke(input={'input': 'Assess the risk of the project and provide a recommendation on how to improve the project\'s risk profile.'})

        return response.content


    def create_payload_dict(self, payload: dict, risk_bootstrap: dict) -> dict:
        """
        Create the payload dictionary for the LLM

        Parameters:
        -----------
        dict:
            The payload with the risk assessment results
        dict:
            The bootstrap results of the risk assessment

        Returns:
        --------
        dict
            The payload dictionary for the LLM
        """
        # create the results
        results = {}

        # add the results to the payload
        results['npv'] = payload.get('npv')
        results['msp'] = payload.get('msp')
        results['riskMSP'] = payload.get('riskMSP')
        results['riskNPV'] = payload.get('riskNPV')

        # calculate the confidence intervals for MSP and NPV
        msp_dist = np.percentile(risk_bootstrap.get('msp_dist'), [2.5, 97.5])
        npv_dist = np.percentile(risk_bootstrap.get('vpl_dist'), [2.5, 97.5])

        # calculate the delta for MSP and NPV
        results['msp_max_delta'] = round(msp_dist[1] - self.MSP_GOAL, 2)
        results['msp_min_delta'] = round(msp_dist[0] - self.MSP_GOAL, 2)
        results['npv_max_delta'] = round(npv_dist[1] - self.VPL_GOAL, 2)
        results['npv_min_delta'] = round(npv_dist[0] - self.VPL_GOAL, 2)

        return json.dumps(results)

    def create_prompt(self, payload: dict, risk_bootstrap: dict) -> str:
        """
        Create the prompt for the risk analysis

        Parameters:
        -----------
        dict:
            The payload with the risk assessment results
        dict:
            The bootstrap results of the risk assessment

        Returns:
        --------
        str
            The prompt for the risk analysis
        """
        # create the payload dictionary
        results = self.create_payload_dict(payload, risk_bootstrap)

        # create the prompt
        messages = [
            SystemMessage(self.system_template),
            HumanMessage(self.human_template.replace('#RESULTS#', results))
        ]
        prompt = ChatPromptTemplate.from_messages(messages=messages)

        return prompt
        