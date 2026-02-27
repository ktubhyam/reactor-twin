#!/bin/bash

# ReactorTwin Git Setup and Commit Script
# Run this script to initialize git and commit all Phase 1-4 work

cd /Users/admin/Documents/GitHub/reactor-twin

# Check if git repo exists
if [ ! -d ".git" ]; then
    echo "Initializing git repository..."
    git init
else
    echo "Git repository already exists."
fi

# Add all files
echo "Adding all files..."
git add .

# Commit with comprehensive message
echo "Creating commit..."
git commit -m "feat: ReactorTwin Phases 1-4 Complete

Phase 1 (Foundation):
- Core Neural ODE with adjoint method
- Abstract base classes (AbstractNeuralDE, AbstractReactor, AbstractKinetics, AbstractConstraint)
- Registry system for extensibility
- CSTR reactor + Arrhenius kinetics
- Positivity constraint (hard/soft modes)
- Complete project structure

Phase 2 (Physics Constraints + Training):
- 7 physics constraints: mass balance, energy balance, thermodynamics, stoichiometry, port-Hamiltonian, GENERIC, positivity
- 2 CSTR benchmarks: exothermic A→B, Van de Vusse
- Training infrastructure: Trainer, MultiObjectiveLoss, ReactorDataGenerator

Phase 3 (Advanced Neural DEs):
- Latent Neural ODE (encoder/decoder for high-dim)
- Augmented Neural ODE (extra dimensions)
- Neural SDE (uncertainty quantification)
- Neural CDE (irregular time series)

Phase 4 (Additional Reactors):
- 3 reactors: BatchReactor, SemiBatchReactor, PlugFlowReactor (PFR with Method of Lines)
- 4 kinetics: MichaelisMenten, PowerLaw, LangmuirHinshelwood, Reversible
- 3 benchmarks: Bioreactor, Consecutive (A→B→C), Parallel (A→B, A→C)

Total: 50+ modules with complete type hints and documentation

Co-Authored-By: Claude Sonnet 4.5 <noreply@anthropic.com>"

# Show status
echo ""
echo "Git status:"
git status

echo ""
echo "Recent commits:"
git log --oneline -n 5

echo ""
echo "✅ Done! Now you can:"
echo "   git remote add origin https://github.com/yourusername/reactor-twin.git"
echo "   git push -u origin main"
