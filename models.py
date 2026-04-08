# models.py

from typing import List, Optional, Dict
from pydantic import BaseModel



class Action(BaseModel):
    """Base class for all actions"""
    pass

class Observation(BaseModel):
    """Base class for all observations"""
    done: bool = False
    reward: Optional[float] = None

class State(BaseModel):
    """Base class for all states"""
    episode_id: Optional[str] = None
    step_count: int = 0



class SupplierInfo(BaseModel):
    """
    Represents one supplier in the supply chain.
    Think of this as a single row in a supplier database.
    """
    supplier_id: str          
    name: str                  
    disruption_level: str      
    delay_days: int            
    daily_cost_usd: int       
    products_supplied: List[str]  


class ProductInfo(BaseModel):
    """
    Represents one product affected by disruptions.
    """
    product_id: str            
    name: str                
    revenue_per_unit: int      
    units_at_risk: int        
    days_until_stockout: int   
    affected_by_suppliers: List[str]  


class SupplyChainAction(Action):
    """
    What the AI agent sends to the environment each step.
    The agent looks at the situation and decides what to do
    about ONE supplier at a time.
    """
    supplier_id: str           
    decision: str             
    reasoning: str            


class SupplyChainObservation(Observation):
    """
    What the environment sends back to the AI agent after each action.
    This is everything the agent needs to make its next decision.
    
    Inherits from Observation, so it already has:
    - done: bool
    - reward: Optional[float]
    """
    
    disrupted_suppliers: List[SupplierInfo]   
    affected_products: List[ProductInfo]        
    
    
    total_budget_usd: int                      
    budget_remaining_usd: int                  
    
   
    decisions_made: int                        
    decisions_remaining: int                   
    
   
    last_action_result: str                    
    current_score: float                      
    message: str                               


class SupplyChainState(State):
    """
    Behind-the-scenes tracking of the episode.
    Not shown to the agent directly — used for scoring and logging.
    
    Inherits from State, so it already has:
    - episode_id: Optional[str]
    - step_count: int
    """
    task_id: str = ""             
    total_budget_usd: int = 0       
    budget_remaining_usd: int = 0 
    correct_decisions: int = 0     
    total_decisions: int = 0        
    critical_errors: int = 0        
    final_score: float = 0.0      