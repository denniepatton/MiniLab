"""
Single Analysis Orchestrator - Complex guild-based research workflow

This orchestrator implements a 4-stage process for comprehensive research analysis:
1. Guild leads (Bohr, Feynman, Bayes) create initial plans
2. Each lead consults their guild, guild members discuss together  
3. Non-Bohr leads report to Bohr, Bohr synthesizes
4. Execute plan with delegation, iterate until satisfied

Maximum token budget: 1,000,000 tokens
"""

from __future__ import annotations

import asyncio
import sys
import threading
import select
from pathlib import Path
from typing import Dict, Optional

from MiniLab.agents.base import Agent
from MiniLab.storage.transcript import TranscriptLogger


class InterruptHandler:
    """Handle user interrupts during workflow execution."""
    
    def __init__(self):
        self.interrupt_requested = False
        self.user_message = None
        self.monitoring = False
        self.monitor_thread = None
    
    def start_monitoring(self):
        """Start background thread to monitor for Enter key."""
        if self.monitoring:
            return
        
        self.monitoring = True
        self.interrupt_requested = False
        self.user_message = None
        
        def monitor():
            while self.monitoring:
                # Check if input is available (non-blocking)
                if sys.stdin in select.select([sys.stdin], [], [], 0.5)[0]:
                    try:
                        line = sys.stdin.readline().strip()
                        if line:  # User typed something
                            self.user_message = line
                        self.interrupt_requested = True
                        break
                    except:
                        pass
        
        self.monitor_thread = threading.Thread(target=monitor, daemon=True)
        self.monitor_thread.start()
    
    def stop_monitoring(self):
        """Stop monitoring for interrupts."""
        self.monitoring = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=1.0)
    
    def check_interrupt(self):
        """Check if interrupt was requested. Returns True if interrupted."""
        return self.interrupt_requested
    
    def reset(self):
        """Reset interrupt state."""
        self.interrupt_requested = False
        self.user_message = None


# Guild structure from config
GUILDS = {
    "synthesis_core": {
        "lead": "bohr",
        "members": ["farber", "gould"],
    },
    "theory_core": {
        "lead": "feynman",
        "members": ["shannon", "greider"],
    },
    "implementation_core": {
        "lead": "bayes",
        "members": ["hinton", "dayhoff"],
    },
}


def _show_progress(message: str):
    """Display progress message with overwrite."""
    sys.stdout.write(f"\r{message}")
    sys.stdout.flush()


def _clear_progress():
    """Clear progress message."""
    sys.stdout.write("\r" + " " * 80 + "\r")
    sys.stdout.flush()


async def run_single_analysis(
    agents: Dict[str, Agent],
    research_question: str,
    max_tokens: int = 1_000_000,
    logger: Optional[TranscriptLogger] = None,
) -> Dict:
    """
    Run a comprehensive Single Analysis workflow.
    
    Args:
        agents: All available agents
        research_question: The research question to analyze
        max_tokens: Maximum token budget (default 1M)
        logger: Optional transcript logger
        
    Returns:
        Dict with workflow results and metadata
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    token_count = 0
    TOKEN_ESTIMATE_PER_CALL = 2000  # Conservative estimate
    
    print("\n" + "=" * 80)
    print("SINGLE ANALYSIS WORKFLOW")
    print("=" * 80)
    print(f"Research Question: {research_question}")
    print(f"Token Budget: {max_tokens:,}")
    print("=" * 80 + "\n")
    
    # =========================================================================
    # STAGE 0: Bohr assigns project name
    # =========================================================================
    print("📝 STAGE 0: Assigning Project Name\n")
    
    bohr = agents["bohr"]
    _show_progress("  Bohr is assigning a project name...")
    
    naming_prompt = f"""Given this research question, provide a concise, descriptive project name (2-4 words, CamelCase, no spaces or special characters):

Research Question: {research_question}

Provide ONLY the project name, nothing else. Examples:
- "CancerGenomicsDeepLearning"
- "ProteinStructurePrediction"

Project name:"""
    
    project_name_response = await bohr.arespond(naming_prompt)
    token_count += TOKEN_ESTIMATE_PER_CALL
    
    # Extract project name (first word, clean it up)
    project_name = project_name_response.strip().split()[0]
    # Remove any non-alphanumeric characters
    project_name = "".join(c for c in project_name if c.isalnum())
    
    # Update output path with proper name (append to output_dir, don't go to parent)
    output_path = output_path / project_name
    output_path.mkdir(parents=True, exist_ok=True)
    
    _clear_progress()
    print(f"  ✓ Project name: {project_name}")
    print(f"  ✓ Output directory: {output_path}\\n")
    
    if logger:
        # Update logger session name with proper project name
        logger.update_session_name(f"single_analysis_{project_name}")
        logger.log_agent_response(
            agent_name="Bohr",
            agent_id="bohr",
            message=f"Project name assigned: {project_name}",
            tokens_used=TOKEN_ESTIMATE_PER_CALL
        )
    
    # =========================================================================
    # STAGES 1-3: STREAMLINED - Skip elaborate planning, go straight to execution
    # =========================================================================
    if logger:
        logger.log_stage_transition(
            "Stages 1-3 (Condensed)",
            "Streamlined planning - focus on execution"
        )
    
    print("📋 STAGES 1-3: Comprehensive Planning with Team Collaboration\n")
    print("  This planning phase is CRITICAL - we take time here to avoid wasting tokens later.\n")
    
    # STAGE 1: Get initial plan from Bohr
    _show_progress("  Stage 1: Bohr drafting comprehensive master plan...")
    
    comprehensive_plan_prompt = f"""Research Question: {research_question}
Project Name: {project_name}

You are Bohr, the PI coordinating this research team:
Farber, Gould, Feynman, Shannon, Greider, Bayes, Hinton, Dayhoff

Create a detailed master plan for the analysis:

1. **Team Roles**: Who does what?
   - Statistical design → Bayes/Shannon
   - Code/scripts → Hinton/Dayhoff  
   - Biology → Greider
   - Clinical → Farber
   - Literature → Gould

2. **Data Exploration**: What files to examine, what to look for

3. **Analysis Approach**: Specific methods, tools, tests

4. **Figures**: What visualizations to create (describe each)

5. **Scripts Needed**: What code to write

6. **Summary Content**: Key points to report

Be specific and detailed so execution is efficient."""

    master_plan = await bohr.arespond(comprehensive_plan_prompt)
    token_count += TOKEN_ESTIMATE_PER_CALL * 2  # Longer response, more tokens
    
    _clear_progress()
    print(f"  ✓ Master plan created ({len(master_plan.split())} words)\n")
    
    # STAGE 2: Get feedback from key specialists
    print("  Stage 2: Consulting key specialists for validation...\n")
    
    specialist_feedback = {}
    
    # Consult Bayes on statistical approach
    _show_progress("    Consulting Bayes (statistician) on methods...")
    bayes = agents["bayes"]
    bayes_prompt = f"""Bohr's master plan for analyzing data:

{master_plan}

As the Bayesian statistician and clinical trial expert, review the statistical analysis plan.
Are the proposed tests appropriate? What would you add/change? Be specific about methods."""
    
    specialist_feedback["bayes"] = await bayes.arespond(bayes_prompt)
    token_count += TOKEN_ESTIMATE_PER_CALL
    _clear_progress()
    print(f"    ✓ Bayes provided statistical review")
    
    # Consult Hinton on computational approach
    _show_progress("    Consulting Hinton (CS engineer) on implementation...")
    hinton = agents["hinton"]
    hinton_prompt = f"""Bohr's master plan:

{master_plan}

As the CS/ML engineer, review the computational and scripting approach.
What scripts are needed? What tools/libraries? Any efficiency concerns? Be specific."""
    
    specialist_feedback["hinton"] = await hinton.arespond(hinton_prompt)
    token_count += TOKEN_ESTIMATE_PER_CALL
    _clear_progress()
    print(f"    ✓ Hinton provided implementation review")
    
    # Consult Greider on biological interpretation
    _show_progress("    Consulting Greider (biologist) on mechanisms...")
    greider = agents["greider"]
    greider_prompt = f"""Bohr's master plan:

{master_plan}

As the molecular biologist, what biological mechanisms should we investigate?
What biological interpretations should we prepare for different findings?"""
    
    specialist_feedback["greider"] = await greider.arespond(greider_prompt)
    token_count += TOKEN_ESTIMATE_PER_CALL
    _clear_progress()
    print(f"    ✓ Greider provided biological context\n")
    
    # STAGE 3: Bohr refines plan with feedback
    _show_progress("  Stage 3: Bohr synthesizing feedback into final execution plan...")
    
    synthesis_prompt = f"""Your original master plan:

{master_plan}

Team feedback:

BAYES (Statistician): {specialist_feedback['bayes']}

HINTON (CS Engineer): {specialist_feedback['hinton']}

GREIDER (Biologist): {specialist_feedback['greider']}

Synthesize this feedback into a FINAL, ACTIONABLE execution plan. Keep the detailed structure
but incorporate their expert suggestions. This will guide all subsequent execution."""
    
    final_plan = await bohr.arespond(synthesis_prompt)
    token_count += TOKEN_ESTIMATE_PER_CALL * 2
    
    _clear_progress()
    print(f"  ✓ Final execution plan synthesized\n")
    print(f"  📊 Planning phase used ~{token_count:,} tokens (important investment!)\n")
    
    if logger:
        logger.log_agent_response(
            agent_name="Bohr",
            agent_id="bohr",
            message=final_plan,
            tokens_used=TOKEN_ESTIMATE_PER_CALL * 2
        )
    
    # Use the refined plan for execution
    master_plan = final_plan
    guild_plans = {"comprehensive": {"plan": final_plan}}
    guild_collaborations = {"specialist_feedback": specialist_feedback}
    
    print("  ⚡ Skipping elaborate planning stages - proceeding to execution\n")
    
    # Dummy condition for old Stage 2 code removal
    if False:
        if logger:
            logger.log_stage_transition(
                "Stage 2",
                "Guild leads consult members, members collaborate"
            )
        
        print("👥 STAGE 2: Guild Collaboration (Parallel)\n")
        
        guild_collaborations = {}
        
        # Phase 2a: All guild leads consult their members in parallel
        all_consultation_tasks = {}
        
        for guild_name, guild_info in GUILDS.items():
            lead_id = guild_info["lead"]
            lead_agent = agents[lead_id]
            member_ids = guild_info["members"]
            
            consultation_tasks = {}
            for member_id in member_ids:
                member_agent = agents[member_id]
                
                consult_prompt = f"""Your guild lead ({lead_agent.display_name}) has created this plan for {guild_name}:

{guild_plans[guild_name]['plan']}

Research Question: {research_question}
Project Name: {project_name}

Provide your expert feedback and suggestions. What would you add, modify, or emphasize?"""
                
                consultation_tasks[member_id] = {
                    "task": member_agent.arespond(consult_prompt),
                    "agent": member_agent,
                }
            
            all_consultation_tasks[guild_name] = {
                "lead": lead_agent,
                "tasks": consultation_tasks,
            }
        
        # Execute all consultations in parallel
        _show_progress(f"  All guild consultations running in parallel...")
        
        for guild_name, guild_consult_info in all_consultation_tasks.items():
            member_responses = {}
            
            for member_id, task_info in guild_consult_info["tasks"].items():
                response = await task_info["task"]
                token_count += TOKEN_ESTIMATE_PER_CALL
                member_responses[member_id] = response
                
                if logger:
                    logger.log_agent_consultation(
                        from_agent=guild_consult_info["lead"].display_name,
                        to_agent=task_info["agent"].display_name,
                        question="Feedback on guild plan",
                        response=response,
                        tokens_used=TOKEN_ESTIMATE_PER_CALL
                    )
            
            guild_collaborations[guild_name] = {
                "member_feedback": member_responses,
            }
        
        _clear_progress()
        for guild_name in guild_collaborations.keys():
            print(f"  ✓ {guild_name} consultations complete")
        
        # Phase 2b: Inter-member discussions in parallel
        if token_count < max_tokens:
            discussion_tasks = {}
            
            for guild_name, guild_info in GUILDS.items():
                member_ids = guild_info["members"]
                if len(member_ids) >= 2:
                    m1_id, m2_id = member_ids[0], member_ids[1]
                    m1_agent, m2_agent = agents[m1_id], agents[m2_id]
                    
                    member_responses = guild_collaborations[guild_name]["member_feedback"]
                    
                    discussion_prompt = f"""You and {m2_agent.display_name} are both members of {guild_name}. Your lead's plan is:

{guild_plans[guild_name]['plan']}

Your feedback: {member_responses[m1_id]}
{m2_agent.display_name}'s feedback: {member_responses[m2_id]}

Research Question: {research_question}

Discuss with {m2_agent.display_name}: what are the most important points to emphasize to your guild lead?"""
                    
                    discussion_tasks[guild_name] = {
                        "task": m1_agent.arespond(discussion_prompt),
                        "m1": m1_agent,
                        "m2": m2_agent,
                    }
            
            _show_progress(f"  Inter-member discussions running in parallel...")
            
            for guild_name, discussion_info in discussion_tasks.items():
                discussion = await discussion_info["task"]
                token_count += TOKEN_ESTIMATE_PER_CALL
                
                guild_collaborations[guild_name]["inter_member_discussion"] = discussion
                
                if logger:
                    logger.log_agent_consultation(
                        from_agent=discussion_info["m1"].display_name,
                        to_agent=discussion_info["m2"].display_name,
                        question="Guild discussion",
                        response=discussion,
                        tokens_used=TOKEN_ESTIMATE_PER_CALL
                    )
            
            _clear_progress()
            print(f"  ✓ Inter-member discussions complete")
        
        print()
        
        if token_count >= max_tokens:
            print(f"⚠️  Token budget exhausted during Stage 2")
        
        # =====================================================================
        # STAGE 3: Non-Bohr leads report to Bohr, synthesis (ALSO SKIPPED)
        # =====================================================================
        # Reserve 20% of tokens for Stage 4 execution
        if token_count < max_tokens * 0.80:
            if logger:
                logger.log_stage_transition(
                    "Stage 3",
                    "Guild leads report to Bohr for synthesis"
                )
            
            print("🧠 STAGE 3: Synthesis by Bohr\n")
            
            bohr = agents["bohr"]
            reports_to_bohr = []
            
            # Non-Bohr leads report their refined plans
            for guild_name, guild_info in GUILDS.items():
                if token_count >= max_tokens:
                    break
                
                lead_id = guild_info["lead"]
                if lead_id == "bohr":
                    continue  # Bohr doesn't report to himself
                
                lead_agent = agents[lead_id]
                _show_progress(f"  {lead_agent.display_name} → Bohr...")
                
                report_prompt = f"""You are the lead of {guild_name}. Report to Bohr with your refined plan after consulting your guild.

Original plan:
{guild_plans[guild_name]['plan']}

Guild feedback:
{guild_collaborations[guild_name]['member_feedback']}

Research Question: {research_question}

Summarize your guild's approach and key contributions."""

                report = await lead_agent.arespond(report_prompt)
                token_count += TOKEN_ESTIMATE_PER_CALL
                
                _clear_progress()
                print(f"  ✓ {lead_agent.display_name} reported to Bohr")
                
                reports_to_bohr.append({
                    "guild": guild_name,
                    "lead": lead_id,
                    "report": report,
                })
                
                if logger:
                    logger.log_agent_consultation(
                        from_agent=lead_agent.display_name,
                        to_agent="Bohr",
                        question=f"Guild {guild_name} report",
                        response=report,
                        tokens_used=TOKEN_ESTIMATE_PER_CALL
                    )
            
            # Bohr synthesizes all guild reports
            if token_count < max_tokens * 0.80:
                _show_progress("  Bohr is synthesizing all guild inputs...")
                
                synthesis_prompt = f"""You are coordinating a comprehensive research analysis. All guild leads have reported their plans:

Research Question: {research_question}

Guild Reports:
{chr(10).join([f"{r['guild']} (led by {agents[r['lead']].display_name}):{chr(10)}{r['report']}{chr(10)}" for r in reports_to_bohr])}

Your guild's plan (synthesis_core):
{guild_plans['synthesis_core']['plan']}

Synthesize all inputs into a comprehensive master plan that:
1. Integrates all guilds' contributions
2. Identifies key tasks and delegation needs
3. Specifies outputs: PDF (single page with figures), legends document, academic write-up with citations
4. Outlines execution order

Be specific about who should do what."""

                master_plan = await bohr.arespond(synthesis_prompt)
                token_count += TOKEN_ESTIMATE_PER_CALL
                
                _clear_progress()
                print("  ✓ Bohr synthesized master plan\n")
                
                if logger:
                    logger.log_agent_response(
                        agent_name="Bohr",
                        agent_id="bohr",
                        message=master_plan,
                        tokens_used=TOKEN_ESTIMATE_PER_CALL
                    )
            else:
                master_plan = "Token budget exhausted before synthesis"    # End of skipped Stages 2-3 block
    pass  # Close the "if False:" block
    
    # =========================================================================
    # STAGE 4: Execution with delegation and iteration
    # =========================================================================
    execution_results = []
    
    # Stage 4 always runs (no token check - we saved tokens by skipping 2-3)
    if True:
        if logger:
            logger.log_stage_transition(
                "Stage 4",
                "Execution with delegation and iteration"
            )
        
        print("⚚️  STAGE 4: Execution with Tool Delegation\n")
        
        # Auto-create project sandbox directory to avoid "directory not found" errors
        project_sandbox = Path("Sandbox") / project_name
        project_sandbox_full = Path.cwd() / project_sandbox
        project_sandbox_full.mkdir(parents=True, exist_ok=True)
        print(f"  ✓ Created project workspace: {project_sandbox}\n")
        
        # Bohr executes the master plan with actual tool delegation
        # Import the pi_coordinated_meeting for delegation logic
        from MiniLab.orchestrators.meetings import run_pi_coordinated_meeting
        
        iteration = 0
        satisfied = False
        interrupt_handler = InterruptHandler()
        execution_instruction = None  # Initialize to None, set on first iteration or after user guidance
        user_guidance = None  # Store user guidance between iterations
        
        print(f"  Starting Stage 4 execution loop (token_count={token_count}, max={max_tokens})")
        print(f"  Will continue until deliverables complete or token budget exhausted")
        print(f"  💡 Press Enter at any time to pause and provide guidance")
        print(f"\n  ℹ️  What is an iteration? Each iteration = Bohr coordinates with team members,")
        print(f"     delegates tasks to appropriate experts, and takes concrete actions.\n")
        
        # Start monitoring for interrupts
        interrupt_handler.start_monitoring()
        
        while not satisfied and token_count < max_tokens:
            iteration += 1
            print(f"  Iteration {iteration}:")
            
            # Build execution instruction for this iteration
            if iteration == 1 or user_guidance:
                user_guidance_section = ""
                if user_guidance:
                    user_guidance_section = f"""
🎯 USER GUIDANCE (INCORPORATE THIS):
{user_guidance}

Please incorporate this guidance as you continue the analysis.
{'=' * 80}

"""
                    user_guidance = None  # Clear after using
                
                execution_instruction = f"""EXECUTE THE ANALYSIS

Research Question: {research_question}
Project: Sandbox/{project_name}/

YOUR PLAN:
{master_plan}

CRITICAL WORKFLOW:
1. Explore data (delegate filesystem 'list' to see files)
2. Write analysis.py script
3. **RUN THE SCRIPT** (delegate terminal command to Hinton/Shannon/Bayes)
4. Verify results exist (check for output files)
5. Write figure generation script
6. **RUN IT** to create figures.pdf
7. Write summary.pdf generation script  
8. **RUN IT**

TEAM DELEGATION:
- Hinton/Dayhoff: filesystem (list/read/write), terminal (run Python scripts)
- Shannon/Bayes: terminal (run statistical analysis)
- Others: consult for design/interpretation

FILESYSTEM: list, read, write, create_dir
TERMINAL: python script.py, ls, cat, etc.

DELIVERABLES in Sandbox/{project_name}/:
1. figures.pdf (MUST exist as file)
2. figure_legends.md (MUST exist as file)
3. summary.pdf (MUST exist as file)

Tokens: {token_count:,}/{max_tokens:,} ({100*token_count//max_tokens}%)

DO NOT just write scripts - you MUST RUN them to produce the PDFs!
Say COMPLETE only when all 3 files physically exist."""
            else:
                execution_instruction = f"""Continue the analysis.

Last iteration: {execution_results[-1]['response'] if execution_results else 'None'}

Still need: figures.pdf, figure_legends.md, summary.pdf
Tokens: {token_count:,}/{max_tokens:,}

Keep working toward deliverables. Say COMPLETE when done."""
            
            # Use pi_coordinated_meeting for proper delegation
            result = await run_pi_coordinated_meeting(
                agents=agents,
                pi_agent_id="bohr",
                user_prompt=execution_instruction,
                project_context=f"Project: {project_name}\nOutput: {output_path}\nIteration: {iteration}",
                max_total_tokens=max_tokens - token_count,
                logger=logger,
            )
            
            token_count += result["estimated_tokens"]
            
            execution_results.append({
                "iteration": iteration,
                "response": result["pi_response"],
                "tool_results": result.get("tool_results", []),
                "consultations": result.get("consultations", {}),
            })
            
            # QUICK-FAIL: Check if tool operation failed
            if result.get("failed_quick", False):
                print(f"\n{'=' * 80}")
                print(f"🛑 STAGE 4 FAILED: Tool execution error detected")
                print(f"{'=' * 80}")
                print(f"Iteration: {iteration}")
                print(f"Error: {result.get('error', 'Unknown error')}")
                print(f"\nExecution stopped to prevent wasting tokens.")
                print(f"Please fix the issue and re-run the analysis.")
                print(f"{'=' * 80}\\n")
                
                # Return early with error state
                return {
                    "success": False,
                    "error": result.get("error"),
                    "project_name": project_name,
                    "output_path": str(output_path),
                    "tokens_used": token_count,
                    "failed_at_stage": 4,
                    "failed_at_iteration": iteration,
                }
            
            # Show tool operations
            if result.get("tool_results"):
                for tool_result in result["tool_results"]:
                    if tool_result.get("success"):
                        tool_name = tool_result.get("tool", "")
                        if tool_name == "filesystem":
                            action = tool_result.get("action", "")
                            path = tool_result.get("path", "")
                            if action == "write":
                                print(f"    ✓ Tool: wrote {path}")
                            elif action == "list":
                                print(f"    ✓ Tool: listed {path}")
                            elif action == "read":
                                print(f"    ✓ Tool: read {path}")
                            else:
                                print(f"    ✓ Tool: {action} {path}")
                        elif tool_name == "terminal":
                            cmd = tool_result.get("command", "")[:50]
                            print(f"    ✓ Tool: ran '{cmd}...'")
                        else:
                            print(f"    ✓ Tool: {tool_name}")
            
            # Check if Bohr indicates completion AND verify deliverables exist
            if "COMPLETE" in result["pi_response"].upper():
                # Verify required deliverables exist in Sandbox (not Outputs)
                sandbox_path = Path.cwd() / "Sandbox" / project_name
                required_files = [
                    sandbox_path / "figures.pdf",
                    sandbox_path / "figure_legends.md",
                    sandbox_path / "summary.pdf",
                ]
                
                missing_files = [f for f in required_files if not f.exists()]
                
                if missing_files:
                    print(f"    ⚠️  Completion claimed but missing required files in Sandbox:")
                    for f in missing_files:
                        print(f"      - Sandbox/{project_name}/{f.name}")
                    
                    # Check if scripts exist but haven't been run
                    script_files = list(sandbox_path.glob("*.py"))
                    if script_files:
                        print(f"    💡 HINT: {len(script_files)} Python script(s) exist but PDFs don't.")
                        print(f"    💡 You MUST run these scripts using terminal tool!")
                        print(f"    💡 Example: DELEGATE to Hinton with terminal command 'cd Sandbox/{project_name} && python script.py'")
                    print(f"    Continuing execution...")
                else:
                    # Copy deliverables to Outputs
                    import shutil
                    print(f"    ✓ All deliverables found in Sandbox")
                    print(f"    📦 Copying to Outputs/{project_name}/...")
                    for src_file in required_files:
                        dest_file = output_path / src_file.name
                        shutil.copy2(src_file, dest_file)
                        print(f"      ✓ Copied {src_file.name}")
                    
                    satisfied = True
                    print(f"    ✓ Analysis complete - all deliverables verified and copied to Outputs")
            else:
                print(f"    ✓ Progress made")
            
            # Check token budget and warn at milestones
            progress_pct = (token_count / max_tokens) * 100
            
            # Warn at 40%, 60%, 80%, 95%
            if 39 < progress_pct < 41:
                print(f"\n  ⚠️  TOKEN WARNING: 40% of budget used ({token_count:,} / {max_tokens:,})")
                print(f"      If no analysis scripts exist yet, you MUST start writing and running them NOW!\n")
            elif 59 < progress_pct < 61:
                print(f"\n  ⚠️  TOKEN WARNING: 60% of budget used ({token_count:,} / {max_tokens:,})")
                print(f"      You should be generating figures and results by now!\n")
            elif 79 < progress_pct < 81:
                print(f"\n  ⚠️  TOKEN WARNING: 80% of budget used ({token_count:,} / {max_tokens:,})")
                print(f"      Focus on completing deliverables - stop exploring, start finalizing!\n")
            elif progress_pct >= 95:
                print(f"\n  🚨 CRITICAL: 95% of token budget used ({token_count:,} / {max_tokens:,})")
                print(f"      Finalize deliverables immediately or analysis will be incomplete!\n")
            
            # Update progress indicator every 3 iterations
            if iteration % 3 == 0:
                print(f"  📊 Progress: Iteration {iteration} | Tokens: {token_count:,}/{max_tokens:,} ({progress_pct:.1f}%)")
            
            # Check for user interrupt
            if interrupt_handler.check_interrupt():
                interrupt_handler.stop_monitoring()
                
                print(f"\n{'=' * 80}")
                print(f"⏸️  PAUSED - User Interrupt")
                print(f"{'=' * 80}")
                
                # Generate status summary
                sandbox_path = Path.cwd() / "Sandbox" / project_name
                files_created = list(sandbox_path.glob("*")) if sandbox_path.exists() else []
                
                deliverables_status = []
                required_files = ["figures.pdf", "figure_legends.md", "summary.pdf"]
                for req_file in required_files:
                    exists = (sandbox_path / req_file).exists()
                    status = "✓" if exists else "○"
                    deliverables_status.append(f"{status} {req_file}")
                
                print(f"\nCurrent Status:")
                print(f"  • Iteration: {iteration}")
                print(f"  • Tokens used: {token_count:,} / {max_tokens:,} ({progress_pct:.1f}%)")
                print(f"  • Files in workspace: {len(files_created)}")
                print(f"  • Deliverables:")
                for status in deliverables_status:
                    print(f"    {status}")
                
                # Show last action
                if result.get("pi_response"):
                    last_action = result["pi_response"][:200]
                    print(f"\n  • Last action: {last_action}...")
                
                print(f"\n{'=' * 80}")
                
                # Get user input
                if interrupt_handler.user_message:
                    user_input = interrupt_handler.user_message
                    print(f"\nYour message: {user_input}\n")
                else:
                    print("\nOptions:")
                    print("  1. Press Enter to continue")
                    print("  2. Type guidance/constraints and press Enter")
                    print("  3. Type 'stop' to end analysis\n")
                    user_input = input("> ").strip()
                
                if user_input.lower() == 'stop':
                    print("\n  Stopping analysis...\n")
                    break
                elif user_input and user_input.lower() != 'continue':
                    # User provided guidance - store it for next iteration
                    user_guidance = f"""{user_input}

Current Status (Iteration {iteration}):
- Tokens used: {token_count:,} / {max_tokens:,}
- Deliverables: {', '.join(deliverables_status)}
"""
                    print(f"\n  Adding your guidance to next iteration...\n")
                else:
                    print(f"\n  Continuing...\n")
                
                # Reset interrupt handler and restart monitoring
                interrupt_handler.reset()
                interrupt_handler.start_monitoring()
        
        # Stop monitoring when loop exits
        interrupt_handler.stop_monitoring()
        
        # Final status check
        if not satisfied:
            if token_count >= max_tokens:
                print(f"\n  ⚠️  Token budget exhausted ({token_count:,} / {max_tokens:,})")
                print(f"  Analysis incomplete - consider continuing with more tokens")
            else:
                print(f"\n  ⚠️  Execution stopped but deliverables not verified")
        
        print()
    
    # =========================================================================
    # Final Summary
    # =========================================================================
    print("=" * 80)
    print("WORKFLOW COMPLETE")
    print("=" * 80)
    print(f"Total Tokens Used: ~{token_count:,} / {max_tokens:,}")
    print(f"Output Directory: {output_path}")
    print("=" * 80 + "\n")
    
    return {
        "research_question": research_question,
        "project_name": project_name,
        "output_dir": str(output_path),
        "token_count": token_count,
        "stages_completed": {
            "stage_1_plans": len(guild_plans),
            "stage_2_collaborations": len(guild_collaborations),
            "stage_3_synthesis": master_plan is not None,
            "stage_4_iterations": len(execution_results),
        },
        "guild_plans": guild_plans,
        "master_plan": master_plan if token_count < max_tokens else None,
        "execution_results": execution_results,
    }
