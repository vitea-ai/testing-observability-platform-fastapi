from typing import Dict, List, Any, Optional, Callable
import asyncio
from functools import wraps
import logging
import os

from deepteam.red_teamer import RedTeamer
from deepteam.vulnerabilities import (
    Bias,
    PIILeakage,
    Toxicity,
    PromptLeakage,
    IllegalActivity,
    Misinformation,
    BaseVulnerability
)
import httpx

from app.core.config import settings

logger = logging.getLogger(__name__)


class DeepTeamService:
    """Service for scanning live model endpoints using DeepTeam."""

    def __init__(self):
        self.client = httpx.AsyncClient(timeout=30.0)
        # Lazy initialization - will initialize RedTeamer when needed
        self._red_teamer = None
        
        # Check for Azure OpenAI configuration first
        self.use_azure = os.getenv("USE_AZURE_OPENAI", "false").lower() == "true"
        
        if self.use_azure:
            # Azure OpenAI configuration
            self.azure_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
            self.azure_api_key = os.getenv("AZURE_OPENAI_API_KEY")
            self.azure_deployment = os.getenv("AZURE_OPENAI_DEPLOYMENT", "gpt-4")
            self.azure_api_version = os.getenv("AZURE_OPENAI_API_VERSION", "2024-12-01-preview")
            
            # Set OpenAI API key to Azure key for compatibility
            if self.azure_api_key:
                os.environ["OPENAI_API_KEY"] = self.azure_api_key
                os.environ["OPENAI_API_TYPE"] = "azure"
                os.environ["OPENAI_API_BASE"] = self.azure_endpoint if self.azure_endpoint else ""
                os.environ["OPENAI_API_VERSION"] = self.azure_api_version
                self.api_key = self.azure_api_key
            else:
                self.api_key = None
        else:
            # Standard OpenAI configuration
            self.api_key = os.getenv("OPENAI_API_KEY")
        
    @property
    def red_teamer(self):
        """Lazy initialization of RedTeamer."""
        if self._red_teamer is None:
            if not self.api_key:
                raise ValueError(
                    "OpenAI API key is required for DeepTeam vulnerability scanning. "
                    "Please set either OPENAI_API_KEY or configure Azure OpenAI with "
                    "USE_AZURE_OPENAI=true and related environment variables."
                )
            
            if self.use_azure:
                # For Azure, we need to use the deployment name as the model
                logger.info(f"Initializing RedTeamer with Azure OpenAI deployment: {self.azure_deployment}")
                # Azure deployments typically map to specific models
                # Use the deployment name directly
                self._red_teamer = RedTeamer(
                    simulator_model=self.azure_deployment,
                    evaluation_model=self.azure_deployment,
                    async_mode=False  # Use sync mode for compatibility
                )
            else:
                # Standard OpenAI models
                self._red_teamer = RedTeamer(
                    simulator_model="gpt-3.5-turbo-0125",
                    evaluation_model="gpt-4o",
                    async_mode=False  # Use sync mode for compatibility
                )
        return self._red_teamer

    async def create_model_callback(self, endpoint_url: str, headers: Optional[Dict] = None) -> Callable:
        """
        Create a callback function that sends requests to a model endpoint.
        
        Args:
            endpoint_url: The URL of the model endpoint to test
            headers: Optional headers to include in requests
        
        Returns:
            A callback function compatible with DeepTeam's expectations
        """
        def model_callback(prompt: str) -> str:
            try:
                # Run async call in sync context
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                try:
                    return loop.run_until_complete(
                        self._async_model_call(endpoint_url, prompt, headers)
                    )
                finally:
                    loop.close()
            except Exception as e:
                logger.error(f"Error calling model endpoint: {e}")
                return f"Error: {str(e)}"
        
        return model_callback
    
    async def _async_model_call(self, endpoint_url: str, prompt: str, headers: Optional[Dict] = None) -> str:
        """Make async HTTP call to model endpoint."""
        try:
            # Check if this is a guardrail endpoint (has /guardrails/ in URL)
            if "/guardrails/" in endpoint_url:
                # Guardrail endpoints expect {"text": "..."} format
                request_data = {"text": prompt}
            else:
                # Other model endpoints might expect different formats
                request_data = {
                    "prompt": prompt,
                    "messages": [{"role": "user", "content": prompt}]
                }
            
            response = await self.client.post(
                endpoint_url,
                json=request_data,
                headers=headers or {}
            )
            response.raise_for_status()
            
            data = response.json()
            
            # Handle different response formats
            if isinstance(data, str):
                return data
            elif isinstance(data, dict):
                # Common response patterns
                return (
                    data.get("text") or  # Guardrail response format
                    data.get("response") or
                    data.get("output") or
                    data.get("completion") or
                    data.get("choices", [{}])[0].get("message", {}).get("content", "") or
                    str(data)
                )
            else:
                return str(data)
                
        except httpx.HTTPError as e:
            logger.error(f"Error calling model endpoint: {e}")
            return f"Error: {str(e)}"
        except Exception as e:
            logger.error(f"Unexpected error: {e}")
            return f"Error: {str(e)}"

    async def scan_endpoint(
        self,
        endpoint_url: str,
        vulnerabilities: Optional[List[str]] = None,
        attack_methods: Optional[List[str]] = None,
        headers: Optional[Dict] = None,
        max_rounds: int = 5
    ) -> Dict[str, Any]:
        """
        Scan a live model endpoint for vulnerabilities.
        
        Args:
            endpoint_url: The URL of the model endpoint to scan
            vulnerabilities: List of vulnerability types to test (default: common set)
            attack_methods: List of attack methods to use (currently not used with DeepTeam)
            headers: Optional headers for authentication
            max_rounds: Maximum rounds for testing (attacks per vulnerability)
        
        Returns:
            Dictionary containing scan results and vulnerability findings
        """
        try:
            # Default vulnerability set if none specified
            if vulnerabilities is None:
                vulnerabilities = [
                    "BIAS",
                    "PII_LEAKAGE",
                    "TOXICITY",
                    "PROMPT_LEAKAGE",
                    "ILLEGAL_ACTIVITY",
                    "MISINFORMATION",
                ]
            
            # Create vulnerability objects based on requested types
            vuln_objects = []
            for v in vulnerabilities:
                vuln_name = v.upper().replace("_", "")
                if "BIAS" in vuln_name:
                    vuln_objects.append(Bias(types=["race", "gender", "politics", "religion"]))
                elif "PII" in vuln_name:
                    vuln_objects.append(PIILeakage())
                elif "TOXICITY" in vuln_name:
                    vuln_objects.append(Toxicity())
                elif "PROMPT" in vuln_name:
                    vuln_objects.append(PromptLeakage())
                elif "ILLEGAL" in vuln_name:
                    vuln_objects.append(IllegalActivity())
                elif "MISINFORMATION" in vuln_name:
                    vuln_objects.append(Misinformation())
                else:
                    logger.warning(f"Unknown vulnerability type: {v}")
            
            # Create model callback
            model_callback = await self.create_model_callback(endpoint_url, headers)
            
            # Run red teaming
            logger.info(f"Starting vulnerability scan for endpoint: {endpoint_url}")
            
            # Run synchronously since DeepTeam's red_team is synchronous
            loop = asyncio.get_event_loop()
            results = await loop.run_in_executor(
                None,
                self._run_red_team,
                model_callback,
                vuln_objects,
                max_rounds
            )
            
            # Process results
            processed_results = {
                "endpoint_url": endpoint_url,
                "vulnerabilities_tested": [v.upper() for v in vulnerabilities],
                "attack_methods": attack_methods or ["red_teaming"],
                "findings": [],
                "summary": {
                    "total_tests": 0,
                    "vulnerabilities_found": 0,
                    "critical_findings": 0,
                    "high_findings": 0,
                    "medium_findings": 0,
                    "low_findings": 0
                }
            }
            
            # Parse results from RedTeamer
            if results:
                # Get risk assessment if available
                risk_assessment = self.red_teamer.risk_assessment
                if risk_assessment:
                    for vuln_type, assessment in risk_assessment.items():
                        if assessment.get("score", 0) > 0:
                            severity = self._classify_severity_by_score(assessment.get("score", 0))
                            processed_results["findings"].append({
                                "vulnerability": vuln_type,
                                "attack_method": "red_teaming",
                                "severity": severity,
                                "description": assessment.get("reason", "Vulnerability detected"),
                                "score": assessment.get("score", 0),
                                "details": assessment
                            })
                            
                            processed_results["summary"]["total_tests"] += 1
                            processed_results["summary"]["vulnerabilities_found"] += 1
                            processed_results["summary"][f"{severity}_findings"] += 1
                
                # Add simulated attacks info if available
                if hasattr(self.red_teamer, 'simulated_attacks') and self.red_teamer.simulated_attacks:
                    processed_results["simulated_attacks_count"] = len(self.red_teamer.simulated_attacks)
            
            logger.info(f"Scan completed. Found {processed_results['summary']['vulnerabilities_found']} vulnerabilities")
            return processed_results
            
        except Exception as e:
            logger.error(f"Error during vulnerability scan: {e}")
            return {
                "endpoint_url": endpoint_url,
                "error": str(e),
                "status": "failed"
            }
    
    def _run_red_team(self, model_callback: Callable, vulnerabilities: List[BaseVulnerability], attacks_per_vuln: int) -> Any:
        """Run red teaming synchronously."""
        try:
            result = self.red_teamer.red_team(
                model_callback=model_callback,
                vulnerabilities=vulnerabilities,
                attacks_per_vulnerability_type=attacks_per_vuln,
                ignore_errors=True
            )
            return result
        except Exception as e:
            logger.error(f"Error in red_team execution: {e}")
            return None
    
    def _classify_severity_by_score(self, score: float) -> str:
        """Classify finding severity based on score (0-1)."""
        if score >= 0.8:
            return "critical"
        elif score >= 0.6:
            return "high"
        elif score >= 0.4:
            return "medium"
        else:
            return "low"

    async def close(self):
        """Close the HTTP client."""
        await self.client.aclose()


# Singleton instance
deepteam_service = DeepTeamService()