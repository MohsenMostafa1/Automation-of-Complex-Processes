import os
import sys
import logging
import subprocess
import yaml
from typing import Dict, Optional, List
from enum import Enum
from dataclasses import dataclass
import shutil
import time
import requests
import json

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('deployment_agent.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class Environment(Enum):
    STAGING = "staging"
    PRODUCTION = "production"

class DeploymentStatus(Enum):
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    SUCCESS = "success"
    FAILED = "failed"
    ROLLED_BACK = "rolled_back"

@dataclass
class DeploymentConfig:
    repo_url: str
    branch: str
    build_command: str
    test_command: str
    deploy_command: Dict[Environment, str]
    health_check_url: Dict[Environment, str]
    rollback_command: Dict[Environment, str]
    required_checks: List[str]
    notify_on_failure: List[str]

class DeploymentAgent:
    def __init__(self, config_path: str = "deployment_config.yaml"):
        self.config = self._load_config(config_path)
        self.current_status = DeploymentStatus.PENDING
        self.deployment_history = []
        self.workspace_dir = "workspace"
        
        # Ensure workspace directory exists
        os.makedirs(self.workspace_dir, exist_ok=True)
    
    def _load_config(self, config_path: str) -> DeploymentConfig:
        """Load deployment configuration from YAML file"""
        try:
            with open(config_path, 'r') as f:
                config_data = yaml.safe_load(f)
                
            return DeploymentConfig(
                repo_url=config_data['repo_url'],
                branch=config_data['branch'],
                build_command=config_data['build_command'],
                test_command=config_data['test_command'],
                deploy_command={
                    Environment.STAGING: config_data['deploy_commands']['staging'],
                    Environment.PRODUCTION: config_data['deploy_commands']['production']
                },
                health_check_url={
                    Environment.STAGING: config_data['health_check_urls']['staging'],
                    Environment.PRODUCTION: config_data['health_check_urls']['production']
                },
                rollback_command={
                    Environment.STAGING: config_data['rollback_commands']['staging'],
                    Environment.PRODUCTION: config_data['rollback_commands']['production']
                },
                required_checks=config_data.get('required_checks', []),
                notify_on_failure=config_data.get('notify_on_failure', [])
            )
        except Exception as e:
            logger.error(f"Failed to load configuration: {str(e)}")
            raise
    
    def _run_command(self, command: str, cwd: Optional[str] = None) -> bool:
        """Execute a shell command and return success status"""
        try:
            logger.info(f"Executing command: {command}")
            process = subprocess.Popen(
                command,
                shell=True,
                cwd=cwd or os.getcwd(),
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                universal_newlines=True
            )
            
            # Stream output in real-time
            while True:
                output = process.stdout.readline()
                if output == '' and process.poll() is not None:
                    break
                if output:
                    logger.info(output.strip())
            
            # Check for errors
            stderr = process.stderr.read()
            if stderr:
                logger.error(stderr.strip())
            
            return process.returncode == 0
        except Exception as e:
            logger.error(f"Command execution failed: {str(e)}")
            return False
    
    def _clone_repository(self) -> bool:
        """Clone or update the git repository"""
        repo_dir = os.path.join(self.workspace_dir, "repo")
        
        if os.path.exists(repo_dir):
            # Update existing repository
            logger.info(f"Updating repository at {repo_dir}")
            return self._run_command(f"git pull origin {self.config.branch}", cwd=repo_dir)
        else:
            # Clone new repository
            logger.info(f"Cloning repository to {repo_dir}")
            return self._run_command(
                f"git clone -b {self.config.branch} {self.config.repo_url} repo",
                cwd=self.workspace_dir
            )
    
    def _run_quality_checks(self) -> bool:
        """Run all pre-deployment quality checks"""
        logger.info("Running pre-deployment quality checks")
        
        # 1. CUDA Compatibility Check
        if "cuda_check" in self.config.required_checks:
            logger.info("Running CUDA compatibility check")
            if not self._run_command("python -c 'import torch; print(torch.cuda.is_available())'", 
                                   cwd=os.path.join(self.workspace_dir, "repo")):
                logger.error("CUDA compatibility check failed")
                return False
        
        # 2. Quantization Check
        if "quantization_check" in self.config.required_checks:
            logger.info("Running quantization check")
            if not self._run_command("python scripts/check_quantization.py", 
                                   cwd=os.path.join(self.workspace_dir, "repo")):
                logger.error("Quantization check failed")
                return False
        
        # 3. MCP + Function Calling Check
        if "function_calling_check" in self.config.required_checks:
            logger.info("Running function calling check")
            if not self._run_command("python tests/test_function_calling.py", 
                                   cwd=os.path.join(self.workspace_dir, "repo")):
                logger.error("Function calling check failed")
                return False
        
        # 4. Cognitive & Agentic RAG Check
        if "rag_check" in self.config.required_checks:
            logger.info("Running RAG check")
            if not self._run_command("python tests/test_rag.py", 
                                   cwd=os.path.join(self.workspace_dir, "repo")):
                logger.error("RAG check failed")
                return False
        
        # 5. Retrieval Check
        if "retrieval_check" in self.config.required_checks:
            logger.info("Running retrieval check")
            if not self._run_command("python tests/test_retrieval.py", 
                                   cwd=os.path.join(self.workspace_dir, "repo")):
                logger.error("Retrieval check failed")
                return False
        
        # 6. Safety Check
        if "safety_check" in self.config.required_checks:
            logger.info("Running safety check")
            if not self._run_command("python tests/test_safety.py", 
                                   cwd=os.path.join(self.workspace_dir, "repo")):
                logger.error("Safety check failed")
                return False
        
        # 7. Human Evaluation Check (simulated)
        if "human_eval_check" in self.config.required_checks:
            logger.info("Running human evaluation check")
            if not self._run_command("python tests/test_human_eval.py", 
                                   cwd=os.path.join(self.workspace_dir, "repo")):
                logger.error("Human evaluation check failed")
                return False
        
        # 8. Optimization Check
        if "optimization_check" in self.config.required_checks:
            logger.info("Running optimization check")
            if not self._run_command("python tests/test_optimization.py", 
                                   cwd=os.path.join(self.workspace_dir, "repo")):
                logger.error("Optimization check failed")
                return False
        
        # 9. Pre-TDD Readiness Check
        if "pre_tdd_check" in self.config.required_checks:
            logger.info("Running pre-TDD readiness check")
            if not self._run_command("python -m pylint --rcfile=.pylintrc src", 
                                   cwd=os.path.join(self.workspace_dir, "repo")):
                logger.error("Pre-TDD readiness check failed")
                return False
        
        return True
    
    def _build_project(self) -> bool:
        """Build the project"""
        logger.info("Building project")
        return self._run_command(self.config.build_command, 
                               cwd=os.path.join(self.workspace_dir, "repo"))
    
    def _run_tests(self) -> bool:
        """Run test suite"""
        logger.info("Running tests")
        return self._run_command(self.config.test_command, 
                               cwd=os.path.join(self.workspace_dir, "repo"))
    
    def _deploy_to_environment(self, environment: Environment) -> bool:
        """Deploy to specified environment"""
        logger.info(f"Deploying to {environment.value}")
        return self._run_command(
            self.config.deploy_command[environment],
            cwd=os.path.join(self.workspace_dir, "repo")
        )
    
    def _health_check(self, environment: Environment) -> bool:
        """Verify deployment health"""
        logger.info(f"Running health check for {environment.value}")
        url = self.config.health_check_url[environment]
        
        try:
            for _ in range(5):  # Retry up to 5 times
                response = requests.get(url, timeout=10)
                if response.status_code == 200:
                    logger.info("Health check passed")
                    return True
                time.sleep(5)
            
            logger.error(f"Health check failed. Status code: {response.status_code}")
            return False
        except Exception as e:
            logger.error(f"Health check error: {str(e)}")
            return False
    
    def _rollback_deployment(self, environment: Environment) -> bool:
        """Rollback deployment"""
        logger.info(f"Rolling back deployment in {environment.value}")
        return self._run_command(
            self.config.rollback_command[environment],
            cwd=os.path.join(self.workspace_dir, "repo")
        )
    
    def _notify_failure(self, error_message: str):
        """Send failure notifications"""
        for recipient in self.config.notify_on_failure:
            logger.info(f"Sending failure notification to {recipient}")
            # In a real implementation, this would send emails, Slack messages, etc.
            # For demo purposes, we'll just log it
            logger.error(f"Notification to {recipient}: {error_message}")
    
    def deploy(self, environment: Environment) -> bool:
        """Execute full deployment pipeline"""
        self.current_status = DeploymentStatus.IN_PROGRESS
        deployment_record = {
            "timestamp": time.time(),
            "environment": environment.value,
            "status": "in_progress",
            "error": None
        }
        
        try:
            # Step 1: Clone/update repository
            if not self._clone_repository():
                raise Exception("Repository clone/update failed")
            
            # Step 2: Run quality checks
            if not self._run_quality_checks():
                raise Exception("Quality checks failed")
            
            # Step 3: Build project
            if not self._build_project():
                raise Exception("Build failed")
            
            # Step 4: Run tests
            if not self._run_tests():
                raise Exception("Tests failed")
            
            # Step 5: Deploy to environment
            if not self._deploy_to_environment(environment):
                raise Exception("Deployment failed")
            
            # Step 6: Health check
            if not self._health_check(environment):
                raise Exception("Health check failed")
            
            # Deployment successful
            self.current_status = DeploymentStatus.SUCCESS
            deployment_record["status"] = "success"
            self.deployment_history.append(deployment_record)
            logger.info("Deployment completed successfully")
            return True
            
        except Exception as e:
            # Deployment failed - attempt rollback
            logger.error(f"Deployment failed: {str(e)}")
            self.current_status = DeploymentStatus.FAILED
            deployment_record["status"] = "failed"
            deployment_record["error"] = str(e)
            self.deployment_history.append(deployment_record)
            
            # Notify about failure
            self._notify_failure(str(e))
            
            # Attempt rollback
            try:
                if not self._rollback_deployment(environment):
                    logger.error("Rollback failed")
                else:
                    logger.info("Rollback completed")
                    self.current_status = DeploymentStatus.ROLLED_BACK
                    deployment_record["status"] = "rolled_back"
            except Exception as rollback_error:
                logger.error(f"Rollback error: {str(rollback_error)}")
            
            return False

if __name__ == "__main__":
    # Example usage
    agent = DeploymentAgent()
    
    # Deploy to staging first
    logger.info("Starting staging deployment")
    if agent.deploy(Environment.STAGING):
        # If staging succeeds, deploy to production
        logger.info("Staging deployment successful. Starting production deployment")
        agent.deploy(Environment.PRODUCTION)
    else:
        logger.error("Staging deployment failed. Aborting production deployment")
