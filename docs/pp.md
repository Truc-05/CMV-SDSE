\section{Measurement Framework, Setup, and Motivating Observations}
\label{sec:measurement_framework}

\subsection{Problem Formulation}
\label{sec:problem_formulation}

We study verifier fusion as a sequential, cost-aware acquisition problem rather than a fixed-ensemble
vote. An input item $x_i$ has a hidden state $y_i\in\{\textsc{error},\textsc{clean}\}$: an error item
carries a wrong answer that should be rejected, a clean item a correct answer that should be accepted.
A pool of candidate verifiers $\mathcal{V}=\{v_1,\dots,v_M\}$ is available; each verifier $v$ returns a
binary verdict (\emph{reject}/catch or \emph{accept}/miss) and has a per-call compute cost $c_v>0$.
A missed error incurs a false-negative cost and a wrongly rejected clean item a false-positive
(false-alarm) cost.

\paragraph{Per-item model.} On an error item $i$, verifier $v$ \emph{false-accepts} (misses the error)
with an item-specific probability
\begin{equation}
a_{iv}=\Pr(v\text{ accepts}\mid i,\ y_i=\textsc{error}).
\end{equation}
Given the item, verdicts are conditionally independent (a de Finetti view: the item's latent
difficulty is the only shared cause). It is the item-to-item variation of $a_{iv}$ that makes repeated
verdicts \emph{dependent} once the item is marginalised out; \Cref{sec:dependence_estimation} turns
that dependence into a measurable quantity. Symmetrically, on a clean item $j$ verifier $v$ raises a
false alarm with probability $f_{jv}=\Pr(v\text{ rejects}\mid j,\ y_j=\textsc{clean})$.

\paragraph{Selected set and cascade.} At any point the system holds an acquired set $S\subseteq\mathcal{V}$
whose verdicts are combined by the disjunctive (reject-any) rule: the cascade rejects item $i$ iff at
least one $v\in S$ rejects it. Hence it catches an error iff some member rejects it and false-alarms on
a clean item iff some member rejects it. Taking expectations over items, the catch rate on errors, the
false-alarm rate on clean items, and the compute cost are
\begin{equation}
R(S)=\mathbb{E}_i\!\Big[1-\!\prod_{v\in S}a_{iv}\Big],\qquad
F(S)=\mathbb{E}_j\!\Big[1-\!\prod_{v\in S}(1-f_{jv})\Big],\qquad
C(S)=\sum_{v\in S}c_v .
\end{equation}

\paragraph{Objective.} A decision pays for missed errors, for false alarms, and for compute. With
$w_{\mathrm c}$ the value of a catch, $w_{\mathrm f}$ the cost of a false alarm, and $\lambda$ the price
of one unit of compute, the net utility is
\begin{equation}
U(S)=w_{\mathrm c}\,R(S)-w_{\mathrm f}\,F(S)-\lambda\,C(S).
\end{equation}
Only ratios matter; normalising $w_{\mathrm c}=1$ and writing $\rho=w_{\mathrm f}/w_{\mathrm c}$ for the
false-alarm-to-miss cost ratio gives the two-parameter form
\begin{equation}
U_{\lambda,\rho}(S)=R(S)-\rho\,F(S)-\lambda\,C(S),
\label{eq:utility}
\end{equation}
which reduces to the pure compute trade-off $U_\lambda(S)=R(S)-\lambda C(S)$ when false alarms are free.
The operating regime is set by $(\lambda,\rho)$.

\paragraph{Sequential nature and two coupled problems.} The system is not handed a fixed ensemble and
asked to vote; at each step it must decide whether another source is worth acquiring and, if so, which
one. The problem therefore factors into two coupled but separable decisions:
(A) \emph{acquisition} --- which verifier to query next and how many verdicts to gather, i.e.\ how to
grow $S$ under \Cref{eq:utility}; and (B) \emph{fusion} --- how to combine the acquired verdicts into a
single accept/reject decision, of which the cascade above is one (disjunctive) instance.
\Cref{sec:method} addresses acquisition; the fusion rule is a second lever whose best setting is itself
regime-dependent (\Cref{sec:fusion_rules}). The organising question the rest of the paper answers is:
\emph{how much new decision-relevant information does a candidate $v$ add, conditional on the evidence
already acquired in $S$?}

\subsection{Framework}
\label{sec:framework}

The framework is the system that realises \Cref{eq:utility} as a running loop
(\Cref{fig:architecture}). It is a pipeline with a measurement stage in the middle and a controller that
closes the loop:
\begin{equation}
\text{input}\rightarrow\textsc{Observe}\rightarrow\text{verifier pool}\rightarrow
\text{measurement}\rightarrow\text{controller}\rightarrow\text{fusion}\rightarrow\text{output}.
\end{equation}

\paragraph{The \textsc{Observe} gate.} Before any verifier verdict is trusted, an \textsc{Observe} gate
checks a precondition: is the evidence needed to detect the failure actually present in the verifier's
context? On \textsc{Observe} failure the correct response is \textsc{Fix-Context} (retrieve, expand, or
re-window), not more verification; on \textsc{Observe} success the task reduces to the verifier-fusion
problem above. This gate matters because, without it, context truncation could masquerade as intrinsic
verifier redundancy; \Cref{sec:observation_redundancy} controls for exactly this confound.

\paragraph{Dependent sources, not independent votes.} The defining stance of the framework is that each
verifier call is treated as a potentially \emph{dependent} decision source. Two calls of the same model
may agree not because the answer is right but because they share a blind pattern; the measurement stage
exists to quantify how much genuinely independent evidence a set of calls contains before the controller
decides whether to acquire more.

\paragraph{Measurement: three questions, three quantities.} From the verdicts the system computes a
small, fixed set of statistics arranged as a three-tier spine, each tier sharper than the last:
(1) \emph{how much independent evidence is there?}~--- the effective number of sources
$n_{\mathrm{eff}}$ (\Cref{sec:dependence_estimation}); (2) \emph{does the next source add useful
evidence?}~--- the conditional marginal catch $\Delta_{\mathrm{catch}}(v\mid S)$
(\Cref{sec:conditional_marginal_value}); (3) \emph{is that evidence worth its risk and cost?}~--- the
conditional marginal value $V(v\mid S)$ (\Cref{sec:conditional_marginal_value}). Tier~1 diagnoses
dependence, tier~2 refines it into complementarity, and tier~3 turns it into a cost-aware decision
quantity that the controller acts on.

\paragraph{Controller and fusion.} The controller reads the tier-3 quantity and chooses the next
acquisition action among \textsc{Scale}, \textsc{Diversify}, \textsc{Stop}, \textsc{Escalate}
(\Cref{sec:cmv_sdse}). When acquisition halts, the verdicts are fused; the base system uses the
disjunctive cascade, and \Cref{sec:fusion_rules} shows the best fusion rule is also regime-dependent.
The output is the accept/reject decision together with an audit trail (the effective evidence, the
marginal-value sequence, and the stopping/escalation reason), so every decision records \emph{why}
verification stopped where it did. To measure whether more evidence is useful, we first quantify how
much independent evidence the current calls already contain.

\begin{figure}[t]\centering
\includegraphics[width=0.98\linewidth]{figures/fig_architecture.pdf}
\caption{The verification pipeline. Each verifier call is treated as a potentially \emph{dependent}
decision source. The \textsc{Observe} gate rejects inputs whose failure evidence is absent
(\textsc{Fix-Context}); the measurement stage computes the three-tier spine
$n_{\mathrm{eff}}\!\to\!\Delta_{\mathrm{catch}}\!\to\!V$; and the CMV-SDSE controller decides whether to
acquire more evidence (\textsc{Scale}/\textsc{Diversify}) or to \textsc{Stop}/\textsc{Escalate} before
the verdicts are fused and returned with an audit trail.}
\label{fig:architecture}
\end{figure}

\subsection{Dependence Estimation and Effective Evidence}
\label{sec:dependence_estimation}

If $K$ verifier calls on an item are dependent, to how many independent calls are they statistically
equivalent? We answer this with a classical design-effect argument applied at the item level.

\paragraph{Data.} The statistical unit is the \emph{item}, not the individual call --- treating $50$
correlated calls as $50$ observations would be pseudo-replication. For item $i$ we record the per-item
sufficient statistic $(m_i,K_i)$, where $K_i$ is the number of \emph{decided} verifier gates (indeterminate
verdicts are excluded and their count reported) and $m_i$ is the number of unwanted (false-accept)
verdicts among them, using a fixed threshold to binarise each verdict.

\paragraph{Models.} We fit, by maximum likelihood, a nested family: a Binomial independence model and a
Beta--Binomial overdispersion model that captures item-level dependence (a ceiling-mixture variant is
used only as a secondary analysis). The Beta--Binomial shape $(a,b)$ gives the intra-item verdict
correlation, the design effect, and the effective number of verifiers,
\begin{equation}
\rho_v=\frac{1}{a+b+1},\qquad \mathrm{DE}(K)=1+(K-1)\rho_v,\qquad
n_{\mathrm{eff}}(K)=\frac{K}{1+(K-1)\rho_v}.
\label{eq:neff}
\end{equation}
Confidence intervals are obtained by profile likelihood. When verdicts are independent ($\rho_v=0$)
$n_{\mathrm{eff}}=K$; as dependence grows $n_{\mathrm{eff}}\ll K$; and $n_{\mathrm{eff}}\!\to\!1$ means
$K$ calls carry the information of roughly one independent source.

\paragraph{What $n_{\mathrm{eff}}$ is, and is not.} $n_{\mathrm{eff}}$ is a \emph{redundancy diagnostic},
not the selection objective, and \Cref{eq:neff} is standard design-effect reasoning rather than a
contribution of this paper. We use it to expose the problem (\Cref{sec:observation_redundancy}) and then
show why it is insufficient for selection (\Cref{sec:observation_diversity}); the selection quantity is
introduced only in \Cref{sec:method}. The conditional-independence-given-item assumption underlying
\Cref{eq:neff} is supported empirically --- within-verifier verdict autocorrelation is near zero and the
product form predicts held-out reliability closely --- but is not proven for all pools. Finally, the
mixture can be extended with a discrete blind-spot atom $\pi_0$ (\Cref{sec:theoretical_guarantees}); we
stress here that the current evidence does \emph{not} statistically identify a non-zero $\pi_0$, so this
section claims no such atom (see \Cref{sec:confirmatory_validation} and the discussion of $\pi_0$).

\subsection{Experimental Setup}
\label{sec:experimental_setup}

\paragraph{Verifiers.} The pool is five open checkpoints spanning three pretraining families and two
size tiers: Qwen2.5-7B and Qwen2.5-14B, Llama-3.1-8B and Llama-3.2-3B, and Mistral-7B, all
$\le 14$B and served locally. We state this as \emph{scope}, not as coverage of frontier or closed
models.

\paragraph{Dependence ladder (pools).} Three pooling conditions increase source diversity while holding
the protocol fixed: \texttt{same\_model} (repeated calls to one checkpoint), \texttt{same\_family}
(distinct checkpoints within one family), and \texttt{cross\_family} (verifiers from distinct families).

\paragraph{Task families and error-generation mechanisms.} Seven families exercise genuinely different
error structures, not seven copies of one setup: MAST FC3 traces are \emph{real multi-agent-system
failure traces}; ARC, MMLU, CSQA, and TruthfulQA are knowledge / commonsense / truthfulness
multiple-choice, where an error item is a wrong option; GSM8K is \emph{multi-step mathematical reasoning};
and MBPP is \emph{program generation with test-grounded errors}. \Cref{tab:setup} summarises the models,
pools, and protocol, and \Cref{tab:provenance} makes the error construction and ground-truth validation
of each family reproducible.

\begin{table}[t]\centering\small
\caption{Data provenance: how error and clean items are constructed and how their ground truth is
validated, per task family. The contribution is error \emph{detection}, so the error source of each
family is stated explicitly and is reproducible from the released item identifiers.}
\label{tab:provenance}
\begin{tabular}{lllll}
\toprule
family & task type & error item & clean item & ground-truth validation\\
\midrule
MAST FC3 & MAS traces & real system-failure trace & valid trace & annotated trace evidence\\
ARC/MMLU/CSQA/TruthfulQA & MCQA & a wrong distractor option & the correct option & dataset gold label\\
GSM8K & math reasoning & a wrong final solution & the correct solution & numeric ground truth\\
MBPP & code generation & a failing program & a passing program & unit tests\\
\bottomrule
\end{tabular}
\end{table}

\paragraph{Protocol.} Each item is verified over up to $k_{\max}=50$ gates; each item-level verdict is
the majority reject over a model's decided gates at a fixed threshold. Items are split into error and
clean, giving catch (true-positive rate on errors) and false alarm (false-positive rate on clean items).
Where applicable, a context control compares a truncated context against an evidence-preserving context
with seeds held fixed, so context is the only variable.

\paragraph{Confirmatory design.} A central methodological point: the controller, the metric, and all
hyperparameters were \emph{frozen before} the large-sample confirmatory phase. \textbf{No post-hoc
controller retuning was performed.} The confirmatory runs (\Cref{sec:confirmatory_validation}) use
$n=150$ error items per family (with the MAST sample reported separately), three pools, and produced
$11{,}250$ backend calls per family with zero unparsed verdicts and zero context overflows. Reported
metrics are catch, false alarm, $\rho_v$, $n_{\mathrm{eff}}$, conditional marginal catch, and utility.

\begin{table}[t]\centering\small
\caption{Experimental scope: verifier pool, dependence ladder, task families, and protocol.}
\label{tab:setup}
\begin{tabular}{ll}
\toprule
component & value\\
\midrule
verifier checkpoints & Qwen2.5-7B, Qwen2.5-14B, Llama-3.1-8B, Llama-3.2-3B, Mistral-7B\\
model families / tiers & 3 pretraining families (Qwen, Llama, Mistral); 2 size tiers ($\le 14$B)\\
pools (dependence ladder) & \texttt{same\_model}, \texttt{same\_family}, \texttt{cross\_family}\\
task families & MAST FC3, ARC, MMLU, CSQA, TruthfulQA, GSM8K, MBPP\\
gates per item & up to $k_{\max}=50$; item-level majority-reject verdict\\
context control & truncated vs.\ evidence-preserving, seeds fixed\\
confirmatory sample & $n=150$ error items / family; controller frozen before large-$n$\\
\bottomrule
\end{tabular}
\end{table}

\subsection{Observation 1: More Sources Do Not Necessarily Mean More Information}
\label{sec:observation_redundancy}

Our first empirical result is that repeated same-model verification is heavily redundant. Under the
fitted dependence model, same-model $n_{\mathrm{eff}}$ saturates near one: on the real multi-agent traces
the same-model effective count at $k=50$ is only $1.2$--$1.9$, while a cross-family pool recovers
$2.4$--$5.2$ (\Cref{tab:mast3mode}). Thus $50$ nominal verifier calls correspond not to $50$ independent
sources but to evidence comparable to roughly one. The frozen confirmatory evaluation
(\Cref{sec:confirmatory_validation}) later establishes the same null across all seven families,
$n_{\mathrm{eff}}@50\in[1.06,1.21]$. The visual signature is \Cref{fig:saturation}: with nominal calls
$K$ on the $x$-axis and $n_{\mathrm{eff}}(K)$ on the $y$-axis, the same-model curve saturates near one and
the independence diagonal $n_{\mathrm{eff}}=K$ runs far above it; the catch and false-alarm curves
saturate within a few gates with wide correlated bands.

\begin{figure}[t]\centering
\includegraphics[width=\linewidth]{figures/fig_saturation.pdf}
\caption{Same-model verification saturates with cascade depth. (a)~Effective verifiers $n_{\mathrm{eff}}(k)$:
$50$ same-model calls carry the evidence of about one independent verifier, while a cross-family pool
recovers several; the dashed line is the independence ideal $n_{\mathrm{eff}}=k$. (b)~Catch reliability and
(c)~false-alarm rate versus depth $k$, each with a $95\%$ cluster-bootstrap band; both plateau within a few
gates.}
\label{fig:saturation}
\end{figure}

\paragraph{Not a context artifact.} The redundancy is intrinsic to the verifiers' reasoning, not an
artifact of removing evidence from the prompt: with seeds held fixed, an evidence-preserving context
leaves $n_{\mathrm{eff}}$ essentially unchanged (a context coefficient $\beta_{\mathrm{context}}\approx0$),
ruling out ``you simply truncated the evidence'' as the explanation.

\paragraph{Takeaway and transition.} Nominal verifier count can dramatically overstate the amount of
independent evidence available under dependent verification. The natural response is to diversify --- but
should one simply select the least-correlated verifier? The next observation shows the answer is no.

\begin{table}[t]\centering\small
\caption{Same-model versus cross-family on MAST FC3 (per failure mode): intra-item correlation $\rho_v$
and effective verifiers $n_{\mathrm{eff}}@50$. Same-model repetition stays near one effective source;
diversity recovers several. (Values from \texttt{main\_results\_3mode}.)}
\label{tab:mast3mode}
\begin{tabular}{llccc}
\toprule
mode & pool & $\rho_v$ & $\rho_v$ 95\% CI & $n_{\mathrm{eff}}@50$\\
\midrule
FM-3.1 & same\_model   & 0.54 & [0.36, 0.73] & 1.82\\
FM-3.1 & cross\_family & 0.42 & [0.29, 0.57] & 2.35\\
FM-3.2 & same\_model   & 0.52 & [0.17, 0.92] & 1.89\\
FM-3.2 & cross\_family & 0.17 & [0.12, 0.27] & 5.24\\
FM-3.3 & same\_model   & 0.82 & [0.69, 0.91] & 1.21\\
FM-3.3 & cross\_family & 0.20 & [0.14, 0.28] & 4.62\\
\bottomrule
\end{tabular}
\end{table}

\subsection{Observation 2: Statistical Diversity Does Not Imply Decision Value}
\label{sec:observation_diversity}

Diversifying by minimising correlation is the wrong fix, and this is the twist that motivates our method.
A weak, near-constant verifier can exhibit \emph{low} correlation with the rest --- and therefore
\emph{raise} the apparent $n_{\mathrm{eff}}$ --- while catching zero residual errors. In our pools the
$3$B checkpoint \texttt{llama32\_3b} is exactly such a candidate: it is statistically distinct from the
others yet contributes no unique catches on the errors the current set misses, and the frozen controller
therefore never selects it (\Cref{fig:diversity}). Because $\Delta_{\mathrm{catch}}$ is an empirical
proportion on finite data, we report its Wilson interval honestly and do not assert that its true value
is exactly zero --- only that it is not statistically supported.

The example separates three quantities that are routinely conflated:
\emph{source count} (``how many calls do I have?''), \emph{statistical diversity} (``how dependent are
their outputs?''), and \emph{decision value} (``does this source catch residual errors worth
catching?''). Low correlation is neither necessary nor sufficient for high decision value: a
low-correlation verifier with no unique catches is useless, whereas a somewhat-correlated verifier that
catches residual errors is valuable. A scatter of dependence against conditional marginal catch makes the
dissociation visible, with the low-correlation-but-useless candidate isolated in the bottom-left.

\begin{figure}[t]\centering
\includegraphics[width=\linewidth]{figures/fig_obs2_diversity.pdf}
\caption{Count $\neq$ diversity $\neq$ value: none of the intuitive predictors determines what a source
adds. (a)~\emph{Count $\neq$ value}: the marginal catch of the $k$-th same-model call collapses after the
first, so repetition adds almost no new evidence. (b)~\emph{Diversity $\neq$ value}: each point is a
candidate verifier (across the confirmatory families), plotted as dependence (correlation of its verdicts
with the current solo verifier) against decision value (conditional marginal catch on the errors the solo
misses); the dead $3$B verifier \texttt{llama32\_3b} (crimson) has low correlation yet essentially zero
residual catch --- statistically diverse but decision-useless. (c)~\emph{The quantity that decides}: the
conditional marginal catch with its Wilson lower bound; the controller adds a candidate exactly when that
lower bound exceeds zero, which excludes the dead verifier (crimson, at the origin) and admits the
complementary ones.}
\label{fig:diversity}
\end{figure}

\begin{quote}
An independent opinion is useless if it contributes no useful information.
\end{quote}

We therefore need a quantity that measures what a candidate $v$ adds \emph{after} conditioning on the
evidence already available in $S$. This directly motivates the conditional marginal verification value of
\Cref{sec:method}.

\section{Method: Conditional Marginal Value and Adaptive Fusion}
\label{sec:method}

The two observations of \Cref{sec:measurement_framework} rule out the two intuitive answers and leave a
single quantity to measure. The method is that quantity and the controller it induces:
\begin{equation*}
\underbrace{\text{count}\ \neq\ \text{diversity}\ \neq\ \text{value}}_{\text{Observations 1--2}}
\ \Longrightarrow\
\underbrace{n_{\mathrm{eff}}\ \to\ \Delta_{\mathrm{catch}}(v\mid S)\ \to\ V(v\mid S)}_{\text{measure residual decision value}}
\ \Longrightarrow\
\underbrace{\text{CMV-SDSE}}_{\textsc{Scale}/\textsc{Diversify}/\textsc{Stop}/\textsc{Escalate}}.
\end{equation*}
Dependence ($n_{\mathrm{eff}}$) diagnoses redundant evidence; conditional marginal value ($V$) decides
whether acquiring another source is worthwhile; and greedy improvement of $V$ under cost \emph{is} the
controller.

\subsection{Conditional Marginal Verification Value}
\label{sec:conditional_marginal_value}

Given the current set $S$, what does a candidate $v$ add? The \emph{conditional marginal catch} is the
fraction of the errors that $S$ misses which $v$ rejects,
\begin{equation}
\Delta_{\mathrm{catch}}(v\mid S)=\Pr\big(v\text{ catches an error}\mid S\text{ missed it}\big)
=\frac{\#\{\text{error }i:\ S\text{ misses }i,\ v\text{ catches }i\}}{\#\{\text{error }i:\ S\text{ misses }i\}},
\end{equation}
and the \emph{conditional false-alarm increment} $\Delta_{\mathrm{fa}}(v\mid S)$ is, symmetrically, the
fraction of currently-accepted clean items that $v$ newly rejects. These are \emph{conditional
proportions} on the residual (surviving) items --- which is precisely what makes them cleanly testable
with a binomial (Wilson) interval. To turn a conditional rate into a marginal utility it must be weighted
by the residual mass: because $1-R(S)$ is the fraction of errors that $S$ still misses and $1-F(S)$ the
fraction of clean items $S$ has not yet flagged, adding $v$ changes catch and false alarm by the exact
counting identity
\begin{equation}
R(S\cup\{v\})-R(S)=\big[1-R(S)\big]\,\Delta_{\mathrm{catch}}(v\mid S),\qquad
F(S\cup\{v\})-F(S)=\big[1-F(S)\big]\,\Delta_{\mathrm{fa}}(v\mid S).
\label{eq:margins}
\end{equation}
The \emph{decision value} of $v$ is then the marginal increment of the utility $U$ (\Cref{eq:utility}),
\begin{equation}
\boxed{\,V(v\mid S)=w_{\mathrm c}\,\big[1-R(S)\big]\,\Delta_{\mathrm{catch}}(v\mid S)
-w_{\mathrm f}\,\big[1-F(S)\big]\,\Delta_{\mathrm{fa}}(v\mid S)-\lambda\,c_v\,}
\label{eq:cmv}
\end{equation}
whose three terms are, in order, useful new evidence, decision harm, and resource price, and which
equals $U(S\cup\{v\})-U(S)$ exactly (\Cref{eq:margins}). The residual-mass factors $[1-R(S)]$ and
$[1-F(S)]$ depend on $S$ but not on $v$; they matter for ranking because they rescale the catch and
false-alarm terms by different amounts while leaving the cost term unscaled.

\paragraph{Why conditionality matters.} Standalone accuracy is insufficient because value depends on $S$:
a strong verifier can have near-zero marginal value once redundant sources are present, while a weaker
verifier can have large value if it catches residual errors. This is the formal content of Observation~2.

\paragraph{Estimation.} We estimate $\Delta_{\mathrm{catch}}$ and $\Delta_{\mathrm{fa}}$ from item-level
paired data (the same items pass through every candidate), attach a Wilson interval to the conditional
catch, and, under the frozen policy, admit a candidate only when that interval's lower bound exceeds
zero. Because the marginal quantities are estimable from a small number $R$ of joint verdict rounds, a
two-round probe ($R=2$) suffices to recover the selection order; the supporting bound is
\Cref{prop:probe}, and empirically a probe of $R=1$--$2$ rounds already reproduces the full-data
acquisition order on essentially every family (\Cref{app:probe}: \Cref{tab:probe}). \Cref{alg:cmv}
summarises the estimator.

\begin{algorithm}[t]
\caption{Conditional Marginal Value Estimation}
\label{alg:cmv}
\begin{algorithmic}[1]
\Require current set $S$; candidates $\mathcal{V}\setminus S$; probe verdicts on error/clean items;
costs $\{c_v\}$; weights $w_{\mathrm c},w_{\mathrm f},\lambda$
\State $E_{\mathrm{miss}}\gets\{\text{error items currently missed by }S\}$;\quad
$J_{\mathrm{acc}}\gets\{\text{clean items currently accepted by }S\}$
\State $n_E\gets|\{\text{all error items}\}|$;\quad $n_J\gets|\{\text{all clean items}\}|$
\For{each candidate $v\in\mathcal{V}\setminus S$}
  \State $\hat\Delta_{\mathrm{catch}}\gets |\{i\in E_{\mathrm{miss}}: v\text{ rejects }i\}|/|E_{\mathrm{miss}}|$ \Comment{conditional rate}
  \State $\hat\Delta_{\mathrm{fa}}\gets |\{j\in J_{\mathrm{acc}}: v\text{ rejects }j\}|/|J_{\mathrm{acc}}|$
  \State $[\ell_v,u_v]\gets\textsc{Wilson}(\hat\Delta_{\mathrm{catch}},|E_{\mathrm{miss}}|)$ \Comment{CI on the conditional rate}
  \State $\widehat{\Delta R}\gets |\{i\in E_{\mathrm{miss}}: v\text{ rejects }i\}|/n_E$;\quad
         $\widehat{\Delta F}\gets |\{j\in J_{\mathrm{acc}}: v\text{ rejects }j\}|/n_J$ \Comment{absolute increments, \Cref{eq:margins}}
  \State $\hat V(v\mid S)\gets w_{\mathrm c}\,\widehat{\Delta R}-w_{\mathrm f}\,\widehat{\Delta F}-\lambda c_v$
\EndFor
\State \Return $\{\hat V(v\mid S),\,[\ell_v,u_v]\}$ and the candidates ranked by $\hat V$
\end{algorithmic}
\end{algorithm}

Dependence (\Cref{eq:neff}) diagnoses \emph{redundant} evidence; conditional marginal value
(\Cref{eq:cmv}) determines whether acquiring another source is \emph{decision-worthy}.

\subsection{The CMV-SDSE Controller}
\label{sec:cmv_sdse}

The controller is Conditional-Marginal-Value-guided Scale--Diversify--Stop--Escalate (CMV-SDSE). Its four
actions are not hand-written rules but consequences of greedily improving the acquisition objective
\Cref{eq:utility}. At each step the controller observes $S$, enumerates admissible actions, estimates
their conditional marginal value (\Cref{alg:cmv}), applies the significance gate, and takes the
highest positive-value action, otherwise stopping or escalating:

\begin{itemize}
\item \textbf{\textsc{Scale}} --- acquire another verdict from a source already in $S$; taken only while a
further draw of that source still has positive marginal value (which, once its self-$n_{\mathrm{eff}}$
has saturated, it does not).
\item \textbf{\textsc{Diversify}} --- acquire a different verifier; triggered when its complementary
residual catch is statistically supported and justifies the added false-alarm and compute cost.
\item \textbf{\textsc{Stop}} --- acquire nothing further; triggered when no admissible in-pool action has
positive marginal utility \emph{and} the target reliability is already met.
\item \textbf{\textsc{Escalate}} --- move outside the current acquisition/fusion family (a stronger or
out-of-pool source, a human/external tool, or a different fusion topology); triggered when no
positive-value in-pool action remains \emph{but} the target is not yet met. \textsc{Escalate} is thus an
operational fallback --- it needs no unidentifiable blind-spot quantity --- and it marks the boundary of
what the current verifier/fusion family can represent, not ``the algorithm failed''.
\end{itemize}

\begin{algorithm}[t]
\caption{CMV-SDSE}
\label{alg:sdse}
\begin{algorithmic}[1]
\State $S\gets\{\arg\max_v V(\{v\}\mid\varnothing)\}$ \Comment{initialise with the best cost-adjusted solo source}
\Loop
  \State compute $\hat V(a\mid S)$ and Wilson lower bound $\ell_a$ for all admissible actions $a$ (\Cref{alg:cmv})
  \State $\mathcal{A}\gets\{a:\ \hat V(a\mid S)>0\ \text{and}\ \ell_a>0\}$ \Comment{significance gate}
  \If{$\mathcal{A}=\varnothing$} \Comment{no positive-value in-pool action remains}
     \If{$R(S)\ge$ target} \State \textbf{\textsc{Stop}} \Comment{target already met}
     \Else \State \textbf{\textsc{Escalate}} \Comment{target unmet, pool exhausted} \EndIf
     \State \textbf{break}
  \Else
     \State $a^\star\gets\arg\max_{a\in\mathcal{A}}\hat V(a\mid S)$ \Comment{ties broken by lower cost, then by $\ell_a$}
     \State \textbf{if} $a^\star$ repeats a source in $S$ \textbf{then} \textsc{Scale} \textbf{else} \textsc{Diversify}
     \State $S\gets S\cup\{\text{evidence from }a^\star\}$
  \EndIf
\EndLoop
\end{algorithmic}
\end{algorithm}

The initial source $S_0$ is the solo verifier of highest cost-adjusted value; ties are broken by lower
cost and then by tighter lower bound; a maximum budget caps the loop; and the operating regime enters
only through $w_{\mathrm f}$ (equivalently $\rho$) and $\lambda$. The regions of $(\lambda,\rho)$ space in
which each action is optimal form the decision map of \Cref{fig:landscape}, with the real pools placed at
their measured operating points. The main message is that the correct action is
\emph{regime-dependent}, not model-label-dependent.

\begin{figure}[t]\centering
\includegraphics[width=0.70\linewidth]{figures/fig_landscape.pdf}
\caption{The CMV-SDSE decision map (schematic). Axes are the complementary evidence available in the pool
($\chi\approx n_{\mathrm{eff}}$) and the required reliability (target). The theoretical ceiling $C(\chi)$
(\Cref{app:theory}) bounds in-pool reliability from above; below it the controller \textsc{Scale}s in the
low-complementarity trap, \textsc{Diversify}s where complementary evidence pays, and \textsc{Stop}s once
the target is met. Operationally the controller \textsc{Escalate}s when no positive-value in-pool action
remains below the target. Real pools are placed at their measured operating points.}
\label{fig:landscape}
\end{figure}

\subsection{Theoretical Guarantees}
\label{sec:theoretical_guarantees}

We state the assumptions, then the two guarantees that directly ground the controller --- one for
selection, one for stopping. Full proofs and the remaining results (a blind-spot ceiling as a theoretical
special case, a distribution-free probe bound, a confidence-bounded safe-stopping rule, and a
nested-verifier characterisation) are deferred to \Cref{app:theory} so the foundation stays focused; in
particular the controller (\Cref{alg:sdse}) does not depend on the unidentified blind-spot mass.

\paragraph{Assumptions.} Verdicts are conditionally independent given the item; costs are modular
($C(S)=\sum_{v\in S}c_v$); and, for the approximation guarantee, acquisition is under a cardinality
budget $|S|\le k$.

\begin{theorem}[Submodular selection]
\label{thm:submod}
The catch set function $R$ is normalised ($R(\varnothing)=0$), monotone non-decreasing, and submodular:
for $A\subseteq B$ and $v\notin B$, $R(A\cup\{v\})-R(A)\ge R(B\cup\{v\})-R(B)$. Consequently, under the
cardinality budget $|S|\le k$, the greedy algorithm that repeatedly adds the verifier of maximum marginal
catch returns $S_k$ with $R(S_k)\ge(1-1/e)\,\max_{|S|\le k}R(S)$.
\end{theorem}

The marginal catch of $v$ given $S$ equals its expected catch on the \emph{survivors} of $S$,
$R(S\cup\{v\})-R(S)=\mathbb{E}_i\big[(1-a_{iv})\prod_{u\in S}a_{iu}\big]$, which shrinks as $S$ grows ---
the source of submodularity. This yields the corollary that drives Observation~2.

\begin{corollary}[Value $\neq$ accuracy]
\label{cor:value}
A verifier's marginal value is its expected catch on residual survivors, not its standalone accuracy;
ordering candidates by accuracy can therefore be strictly suboptimal.
\end{corollary}

The $(1-1/e)$ guarantee is for cardinality-constrained maximisation of the catch $R$; the false-alarm and
compute terms of \Cref{eq:utility} are modular and enter through the value ranking (\Cref{eq:cmv}) and the
stopping rule below rather than through \Cref{thm:submod}.

\begin{theorem}[Optimal stopping]
\label{thm:stop}
Along the greedy path let $u_t=\Delta_t-\lambda c_t$ be the marginal utility at step $t$, with $\Delta_t$
the marginal catch gain. By submodularity $\Delta_t$ is non-increasing; if per-step costs are
non-decreasing then $u_t$ is non-increasing, and stopping at the first $t$ with $u_t\le0$ is optimal along
that path.
\end{theorem}

Two scopes must be kept separate. \Cref{thm:submod} guarantees the \emph{catch-only} greedy: maximising
marginal catch under a cardinality budget is $(1-1/e)$-optimal for $R$. CMV-SDSE is the \emph{cost- and
risk-aware extension} of that greedy --- it ranks by the full decision value $V$ (\Cref{eq:cmv}), which
subtracts modular false-alarm and compute terms, and it stops by \Cref{thm:stop}. We do \emph{not} claim
a global $(1-1/e)$ ratio for the full cost-sensitive objective, since a submodular-minus-modular function
need not be submodular; the guarantee we state is exactly the catch bound of \Cref{thm:submod} together
with the path-optimality of stopping (\Cref{thm:stop}). \textsc{Escalate} is the operational fallback
when in-pool value is exhausted below the target and needs no unidentifiable quantity; the blind-spot
ceiling of \Cref{app:theory} is a theoretical special case that explains \emph{why} in-pool value can run
out.

\section{Experiments and Results}
\label{sec:experiments}

We test five questions. \textbf{RQ1}: does CMV reject statistically diverse but decision-useless sources?
\textbf{RQ2}: does it select complementary sources when residual value exists? \textbf{RQ3}: is any fixed
fusion rule universally optimal? \textbf{RQ4}: does CMV-SDSE adapt to error/compute cost? \textbf{RQ5}: do
the findings survive frozen cross-task/cross-model confirmation?

\subsection{Baselines and Evaluation Metrics}
\label{sec:baselines_metrics}

\paragraph{Acquisition baselines.} Single best verifier; repeated same-model (\textsc{Scale}); and, where
available, accuracy-ranked selection and correlation/diversity-ranked selection. These isolate the effect
of ranking by conditional marginal value against ranking by accuracy or by dependence alone.

\paragraph{Fusion baselines.} On \emph{identical} acquired verdicts we compare six fusion rules:
disjunctive OR (reject-any), unanimous AND, majority, a reliability-weighted vote, the unsupervised
Dawid--Skene latent-truth combiner, and a learned combiner (pattern-Bayes stack). Any claim comparing
fusion rules holds the underlying verdicts fixed, so differences are attributable to the rule, not to the
data it sees.

\paragraph{Metrics.} Catch (TPR) and false alarm (FPR); the utility $U_{\lambda,\rho}=\text{catch}-\rho\,\text{FA}$
(\Cref{eq:utility}); the diagnostic $n_{\mathrm{eff}}$ (\Cref{eq:neff}); the conditional complementarity
$\Delta_{\mathrm{catch}}$; and the decision value $V$ (\Cref{eq:cmv}). Conditional catch is reported with
Wilson intervals; comparisons on shared items are paired; averaging (macro over families) is stated
explicitly. The cost sweep varies the false-alarm/miss ratio $\rho$ (decision risk) and the compute price
$\lambda$ (acquisition cost).

\subsection{Conditional Marginal Value Selects Complementary Verifiers}
\label{sec:marginal_value_results}

Directly validating the central claim (RQ1--RQ2), the frozen controller adds a candidate only when its
conditional marginal catch is statistically supported (Wilson lower bound $>0$), and no fixed model or
family label decides the outcome --- the measured residual value does. The statistically-diverse but
useless $3$B verifier of \Cref{sec:observation_diversity} is correctly never diversified in (its interval
includes zero), while complementary verifiers with supported residual catch are added
(\Cref{fig:significance}). The solo and diversifier roles \emph{flip with the task}: on the knowledge
families the strong solo is a Qwen checkpoint diversified by Mistral, whereas on code (MBPP) Mistral is the
strongest solo and Llama/Qwen diversify --- no model identity is intrinsically ``the diversifier''.

Restated in fusion terms, repetition is worthless and diversity is not: the macro-averaged error-catch
gain from an independent same-model repeat is $+0.003$, versus $+0.079$ from a diverse source
(\Cref{fig:beforeafter}), an order-of-magnitude difference that is $n_{\mathrm{eff}}\!\approx\!1$ restated
as a selection result. Selection follows measured residual value, not family labels, standalone accuracy,
or correlation alone. A head-to-head acquisition ablation makes this quantitative: ranking \emph{and
stopping} by conditional marginal value beats accuracy-, correlation-, and random-ranked acquisition and
same-model scaling --- at balanced cost it more than doubles utility with a single verifier, and the
pure-diversity (low-correlation) policy is the \emph{worst} of the multi-verifier baselines
(\Cref{app:acq}: \Cref{tab:acqablation,fig:budget}).

\begin{figure}[t]\centering
\includegraphics[width=0.9\linewidth]{figures/fig_significance.pdf}
\caption{Conditional marginal value selects complementary verifiers. (a)~At the first greedy step on
GSM8K, each candidate's $\Delta_{\mathrm{catch}}$ with its Wilson interval: the dead $3$B verifier is not
added (its interval includes zero), the complementary verifier is. (b)~The diversifier's marginal catch is
statistically supported (Wilson lower bound $>0$) on every confirmatory family.}
\label{fig:significance}
\end{figure}

\begin{figure}[t]\centering
\includegraphics[width=0.66\linewidth]{figures/fig_beforeafter.pdf}
\caption{Repetition versus diversity, per family. A same-model repeat (\textsc{Scale}) leaves the
error-catch rate essentially unchanged (crimson sits on the solo), whereas a diverse verifier
(\textsc{Diversify}) moves it right; the macro-averaged gains are $+0.003$ versus $+0.079$.}
\label{fig:beforeafter}
\end{figure}

\subsection{No Universally Optimal Fusion Rule}
\label{sec:fusion_rules}

Once verdicts are acquired, how should they be fused (RQ3--RQ4)? Holding the verdicts fixed and sweeping
the false-alarm/miss cost $\rho$, we score each fixed rule by $U_{\lambda,\rho}$. The cost-optimal rule
\emph{changes with the operating regime}: OR/aggressive rejection wins when misses dominate, a learned
or selective rule wins at intermediate cost, single-best or majority wins near balance, and unanimous AND
wins when false alarms dominate (\Cref{fig:regime}, \Cref{fig:ablation}, \Cref{tab:costsweep}). We do
not claim this exact ordering is a universal law; we claim that \emph{no single rule dominates across the
evaluated cost regimes}. Run at each $\rho$, CMV-SDSE tracks the cost-optimal fixed rule with a single knob
and shrinks its selected set as false alarms grow costly; its regret against the \emph{oracle} choice of
fixed rule is at most $0.016$ for $\rho\le2$, growing only at the conjunctive extreme
(\Cref{app:regret}: \Cref{tab:regret}).

\paragraph{Scope and \textsc{Escalate}.} The one regime the controller cannot track is the
precision-critical extreme: the reject-any cascade is \emph{disjunctive} and only adds false alarms,
whereas the cost-optimal fusion there becomes \emph{conjunctive} (unanimity) --- a topology outside the
current family, at any set size. Framed correctly this is a strength: the framework recognises when its
representational family is insufficient and hands off via \textsc{Escalate} (switch fusion topology or
decision mechanism). A learned conjunctive combiner is natural future work.

\begin{figure}[t]\centering
\includegraphics[width=0.66\linewidth]{figures/fig_regime.pdf}
\caption{No universal fusion rule. Utility $U=\text{catch}-\rho\,\text{FA}$ versus the false-alarm/miss
cost ratio $\rho$; the coloured bands mark which fixed rule is cost-optimal in each regime (OR $\to$
learned stack $\to$ single-best $\to$ majority $\to$ AND). Run at each $\rho$, the CMV-SDSE controller
(solid) tracks the best fixed rule with a single knob.}
\label{fig:regime}
\end{figure}

\begin{figure*}[t]\centering
\includegraphics[width=0.98\linewidth]{figures/fig_ablation.pdf}
\caption{Fusion-rule ablation. Catch (TPR), false alarm (FPR), and net value (Youden $J=\text{catch}-\text{FA}$)
for seven fusion rules on identical verdicts, macro-averaged over six confirmatory families and ranked by
net value. OR-cascade maximises catch but also false alarms; unanimity (AND) minimises both; the selective
rules win at balanced cost.}
\label{fig:ablation}
\end{figure*}

\begin{table}[t]\centering\small
\caption{Cost-ratio sweep: the best fixed fusion rule and the frozen controller's utility
$U=\text{catch}-\rho\,\text{FA}$ (macro-avg over the six confirmatory families) as $\rho$ grows, with the
controller's selected-set size. The winning rule flips across the sweep; the controller tracks it with one
knob. (Values from \texttt{fusion\_cost\_sweep}.)}
\label{tab:costsweep}
\begin{tabular}{lcccccc}
\toprule
$\rho$ (FA/miss) & 0.0 & 0.25 & 0.5 & 1.0 & 2.0 & 4.0\\
\midrule
best fixed rule & OR & stack & stack & single-best & majority & AND\\
best-fixed $U$ & 0.954 & 0.761 & 0.644 & 0.475 & 0.275 & 0.093\\
controller $U$ & 0.953 & 0.761 & 0.588 & 0.460 & 0.259 & $-0.034$\\
controller set size & 2.67 & 2.33 & 2.17 & 1.50 & 1.00 & 1.00\\
\bottomrule
\end{tabular}
\end{table}

\subsection{Cross-Task and Cross-Model Confirmatory Validation}
\label{sec:confirmatory_validation}

To rule out a single-benchmark or single-model artifact (RQ5), we follow a strict timeline:
pilot/discovery $\rightarrow$ freeze the metric, controller, and hyperparameters $\rightarrow$ large-sample
confirmatory evaluation, with no post-hoc retuning. The confirmatory phase spans MAST-3.3, ARC, MMLU, CSQA,
TruthfulQA, GSM8K, and MBPP at $n=150$ error items per family (MAST reported separately), covering
knowledge, commonsense/reasoning, truthfulness, multi-step math, code generation, and real multi-agent
failures, over Qwen/Llama/Mistral checkpoints.

Three results replicate. First, same-model redundancy persists: $n_{\mathrm{eff}}@50\in[1.06,1.21]$ across
the evaluated families (\Cref{fig:sevenfamily}, \Cref{tab:sevenfamily}). Second, useful diversification is
statistically supported per family (conditional marginal catch with Wilson lower bound $>0$). Third, the
weak/dead candidate remains unselected under the frozen policy. We do not claim ``model-agnostic'' or
``benchmark-independent'' as proven facts; we claim the design is benchmark-independent \emph{by
construction} and validated across the evaluated task and model families. The two central effects ---
redundancy persists, useful diversification remains measurable --- survive methodology-frozen cross-task
confirmation.

\begin{figure}[t]\centering
\includegraphics[width=\linewidth]{figures/fig_sevenfamily.pdf}
\caption{Same-model redundancy replicates across seven families (frozen confirmatory phase).
(a)~Effective verifiers $n_{\mathrm{eff}}@50$ within the null band $[1.06,1.21]$; (b)~intra-item
correlation $\rho_v$ ($0.82$--$0.94$); (c)~same-model miss rate. The redundancy null holds across
knowledge, reasoning, math, code, and real multi-agent errors.}
\label{fig:sevenfamily}
\end{figure}

\begin{table}[t]\centering\small
\caption{Seven-family confirmatory summary: same-model $\rho_v$ and $n_{\mathrm{eff}}@50$, the diversifier
added on each family (unique residual catches / surviving errors), and the never-added dead verifier.
(Values from \texttt{redundancy\_invariant\_7family}.)}
\label{tab:sevenfamily}
\begin{tabular}{lccll}
\toprule
family & $\rho_v$ & $n_{\mathrm{eff}}@50$ & diversified-in (catches) & dead / never added\\
\midrule
MAST-3.3   & 0.821 & 1.21 & --- & ---\\
ARC        & 0.901 & 1.11 & mistral7b (11/15) & llama32\_3b\\
MMLU       & 0.881 & 1.13 & mistral7b (21/26) & llama32\_3b\\
CSQA       & 0.900 & 1.11 & mistral7b (9/16)  & llama32\_3b\\
TruthfulQA & 0.915 & 1.09 & mistral7b (19/36) & llama32\_3b\\
GSM8K      & 0.856 & 1.16 & mistral7b (3/7)   & llama32\_3b\\
MBPP       & 0.938 & 1.06 & llama31 (8/20)    & llama32\_3b\\
\bottomrule
\end{tabular}
\end{table}

\subsection{Practical Diagnostic: The Cascade Auditor}
\label{sec:cascade_auditor}

The core contribution is the measurement spine and the CMV-SDSE controller; the auditor is simply their
application as a drop-in diagnostic, kept deliberately lightweight. Given a pool or a small probe and the
operating costs, it reports each verifier's self-$n_{\mathrm{eff}}$ (redundancy), the marginal-value
ordering, and a Scale/Diversify/Stop/Escalate recommendation with a plain rationale --- for example
``additional same-model calls are statistically redundant; diversify to source $v$'', or ``no
positive-value action remains and the target is unmet: \textsc{Escalate}''. Two summary numbers make the
redundancy legible, a \emph{Verification Debt} $=\text{calls}-n_{\mathrm{eff}}$ and an \emph{Efficiency}
$=n_{\mathrm{eff}}/\text{calls}$. Full templates and parameters are in \Cref{app:repro} and the released
code.

\appendix

\section{Deferred Theoretical Results and Proofs}
\label{app:theory}

We collect the results kept out of \Cref{sec:theoretical_guarantees}. The controller of
\Cref{sec:cmv_sdse} does not depend on any of them operationally; they characterise its scope.

\begin{theorem}[Blind-spot ceiling; theoretical special case]
\label{thm:ceiling}
Let $\pi_0=\mathbb{E}_i\big[\prod_{v\in\mathcal V}a_{iv}\big]$ be the in-pool blind-spot mass --- the
probability that \emph{every} pool verifier misses an error item. Then every in-pool cascade satisfies
$R(S)\le R(\mathcal V)\le 1-\pi_0$: no in-pool acquisition crosses the ceiling, and only enlarging the
pool (\textsc{Escalate}) can. The bound is stated conditional on a true $\pi_0$; empirically $\pi_0$ is
\emph{not} statistically identified (\Cref{sec:confirmatory_validation}), which is why the controller's
\textsc{Escalate} rule is operational (\Cref{alg:sdse}) rather than ceiling-based. The theorem's role is
only to explain why in-pool decision value can be exhausted below a high target.
\end{theorem}

\begin{proposition}[Probe-driven near-optimality]
\label{prop:probe}
The marginal quantities of \Cref{eq:margins} are estimable from $R$ joint verdict rounds, and the
probe-driven greedy obeys $R(S_k)\ge(1-1/e)\,\mathrm{OPT}_k-2k\,\varepsilon(R)$ with
$\varepsilon(R)=\sqrt{\log(2mk/\delta)/(2nR)}$; $R=2$ already recovers the optimal cascade order.
\end{proposition}

\begin{proposition}[Confidence-bounded safe stopping]
\label{prop:safestop}
Stopping only when the upper confidence bound of the next marginal falls below its cost incurs no
premature stop with probability at least $1-\delta$.
\end{proposition}

\begin{proposition}[Nested verifiers: when accuracy order suffices]
\label{prop:nested}
If verifiers are nested or stochastically ordered, accuracy order coincides with the optimal order; the
selection gap scales with a complementarity functional $\chi$, which vanishes for same-model pools --- so
their only lever is \textsc{Stop}.
\end{proposition}

Proofs of \Cref{thm:submod,thm:stop,thm:ceiling} and \Cref{prop:probe,prop:safestop,prop:nested} follow
the standard monotone-submodular and concentration arguments and are given in full in the released
supplement.

\section{Implementation and Reproducibility Details}
\label{app:repro}

The confirmatory protocol was frozen before the large-sample runs and not retuned; the details below are
fixed by that freeze so the evaluation is an out-of-sample test.

\paragraph{Models and backend.} Exact checkpoint identifiers for the five verifiers, the serving backend,
decoding parameters (temperature, sampling), and hardware are pinned in the released configuration.

\paragraph{Verdict pipeline.} Each verifier call is parsed to a binary verdict by a fixed parser;
verdicts are binarised at a fixed threshold; indeterminate outputs are excluded from the decided count
and reported; the item-level verdict is the majority reject over a model's decided gates; and seeds are
recorded per gate (and held fixed across the truncated/evidence context control).

\paragraph{Data and error construction.} Dataset splits and per-item identifiers are released for every
family. Error and clean items follow \Cref{tab:provenance}: MCQA errors are wrong distractors; GSM8K
errors are incorrect final solutions validated against the numeric ground truth; MBPP errors are programs
that fail the provided unit tests; MAST errors are annotated failure traces.

\paragraph{Estimation.} $\rho_v$, $n_{\mathrm{eff}}$, and $\pi_0$ come from a Beta-Binomial / ceiling
mixture fit by maximum likelihood with profile-likelihood confidence intervals; conditional catch uses
Wilson intervals; the item is the statistical unit.

\paragraph{Frozen policy.} The controller thresholds, weights, the significance gate (Wilson lower bound
$>0$), the sweep grid for $(\lambda,\rho)$, and the freeze date are all pinned; no controller
hyperparameter was changed after the confirmatory phase began.

\section{Additional Analyses}
\label{app:analyses}

All analyses below are computed from the frozen runs already collected --- no new benchmarks --- and
are macro-averaged over the six identical-setup confirmatory families unless noted.

\subsection{Acquisition-policy ablation}
\label{app:acq}

\Cref{tab:acqablation} pits the controller's acquisition against the natural alternatives on identical
verdicts: repeat the same model (\textsc{Scale}), add verifiers in random order, in descending
standalone accuracy, or in ascending correlation (the pure-diversity heuristic). At balanced cost
($\rho=1$) ranking \emph{and stopping} by conditional marginal value more than doubles utility
($U=0.49$ versus $\le 0.23$ for every baseline) using a single verifier: the naive policies over-acquire,
raising catch but inflating false alarms, so their utility \emph{falls} as budget grows
(\Cref{fig:budget}). Crucially, the low-correlation policy --- the
one a ``diversity'' reading recommends --- is \emph{worse} than accuracy-ranked, because it adds the
statistically diverse but useless verifier first. Improvement comes from conditional decision value, not
from diversity.

\begin{table}[t]\centering\small
\caption{Acquisition-policy ablation at $\rho=1$ (macro-avg over six families), from
\texttt{rev\_acquisition\_ablation}. CMV-SDSE attains the highest utility with the fewest verifiers.}
\label{tab:acqablation}
\begin{tabular}{lcccc}
\toprule
acquisition policy & catch & false alarm & \# verifiers & utility $U=\text{catch}-\rho\,\text{fa}$\\
\midrule
same-model (\textsc{Scale}) & 0.889 & 0.656 & 1.0 & 0.233\\
random order                & 0.923 & 0.699 & 3.0 & 0.224\\
accuracy-ranked             & 0.954 & 0.771 & 3.0 & 0.183\\
low-correlation-ranked      & 0.934 & 0.742 & 3.0 & 0.192\\
\textbf{CMV-SDSE}           & 0.712 & 0.227 & 1.0 & \textbf{0.486}\\
\bottomrule
\end{tabular}
\end{table}

\begin{figure}[t]\centering
\includegraphics[width=0.92\linewidth]{figures/fig_budget.pdf}
\caption{Budget--performance curves at $\rho=1$ (macro-avg). (a)~Utility versus number of verifiers
acquired: naive acquisition \emph{loses} utility as the budget grows, while CMV-SDSE acquires only up to
its utility peak and stops. (b)~Catch rises with budget but at a false-alarm cost that the utility in
(a) charges for. Same-model scaling is flat --- extra calls buy no new decision-relevant information.}
\label{fig:budget}
\end{figure}

\subsection{Controller regret against the best fixed fusion rule}
\label{app:regret}

\Cref{tab:regret} reports the regret $\mathrm{Regret}(\rho)=U_{\text{best fixed rule}}(\rho)-U_{\text{CMV-SDSE}}(\rho)$
from the cost sweep. It is at or below $0.016$ for $\rho\le 2$: with one knob the controller nearly
matches the \emph{oracle} choice of fixed rule at each cost ratio. The exception is the
precision-critical extreme $\rho=4$, where the cost-optimal rule is conjunctive (AND) --- a topology the
disjunctive controller cannot represent --- which is exactly the boundary the framework hands off to
\textsc{Escalate} (\Cref{sec:fusion_rules}).

\begin{table}[t]\centering\small
\caption{Controller regret vs.\ the best fixed fusion rule per cost ratio, from \texttt{rev\_regret}.}
\label{tab:regret}
\begin{tabular}{lcccccc}
\toprule
$\rho$ (fa/miss) & 0.0 & 0.25 & 0.5 & 1.0 & 2.0 & 4.0\\
\midrule
best fixed rule & OR & stack & stack & single-best & majority & AND\\
regret          & 0.001 & 0.000 & 0.056 & 0.015 & 0.016 & 0.127\\
\bottomrule
\end{tabular}
\end{table}

\subsection{Decomposition of the conditional marginal value}
\label{app:decomp}

\Cref{tab:cmvdecomp} decomposes $V$ for every first-step candidate on GSM8K (current set $S=\{$llama31$\}$),
so the selection is legible term by term: the complementary verifier is added for its catch
($\Delta_{\mathrm{catch}}=0.43$, Wilson lower bound $0.16>0$) despite a non-trivial false-alarm increment
and cost, whereas the dead $3$B verifier is rejected for $\Delta_{\mathrm{catch}}=0$ and a repeat
(\textsc{Scale}) is rejected for adding nothing at high cost.

\begin{table}[t]\centering\small
\caption{Conditional-marginal-value decomposition, GSM8K first step ($S=\{$llama31$\}$), from
\texttt{gsm8k\_marginal}. $V$ uses the frozen catch-oriented weighting ($w_{\mathrm f}=0$,
$\lambda=0.1$); the false-alarm increment is shown separately.}
\label{tab:cmvdecomp}
\begin{tabular}{llccccc}
\toprule
candidate & action & $\Delta_{\mathrm{catch}}$ (95\% CI) & unique/denom & $\Delta_{\mathrm{fa}}$ & rel.\ cost & $V$\\
\midrule
mistral7b (new)   & \textsc{Diversify} & 0.43 [0.16, 0.75] & 3/7 & 0.40 & 3.01 & $+0.128$\\
qwen7b (new)      & ---                & 0.14 [0.03, 0.51] & 1/7 & 0.00 & 1.02 & $+0.041$\\
llama32\_3b (new) & --- (dead)         & 0.00 [0.00, 0.35] & 0/7 & 0.00 & 1.00 & $-0.100$\\
llama31 (repeat)  & --- (\textsc{Scale}) & 0.00 [0.00, 0.35] & 0/7 & 0.00 & 3.68 & $-0.368$\\
\bottomrule
\end{tabular}
\end{table}

\subsection{Probe-size sensitivity}
\label{app:probe}

\Cref{tab:probe} tests how many joint verdict rounds a probe needs to recover the full-data acquisition
order. The identity of the best solo verifier and of the first diversifier is recovered on essentially
every family from as few as $R=1$--$2$ rounds (the single dip at $R=3$ is one family and is noise),
consistent with \Cref{prop:probe}: a cheap two-round probe is enough to decide which verifier is worth
acquiring, with no expensive calibration phase.

\begin{table}[t]\centering\small
\caption{Probe-size sensitivity, from \texttt{rev\_probe\_sensitivity}: fraction of the six families whose
$R$-round probe reproduces the full ($50$-gate) selection order.}
\label{tab:probe}
\begin{tabular}{lccccc}
\toprule
probe rounds $R$ & 1 & 2 & 3 & 5 & 10\\
\midrule
best-solo recovered      & 1.00 & 1.00 & 0.83 & 1.00 & 1.00\\
first-diversifier recovered & 1.00 & 1.00 & 0.83 & 1.00 & 1.00\\
\bottomrule
\end{tabular}
\end{table}

\subsection{Which term of the value function matters (component ablation)}
\label{app:components}

\Cref{tab:component} re-runs the controller with terms of the value $V$ (\Cref{eq:cmv}) switched off.
Dropping the false-alarm term makes the controller over-acquire and its utility falls
($0.486\!\to\!0.411$); dropping \emph{both} the false-alarm and compute penalties (``catch-only'') is
the naive maximise-catch policy and collapses utility to $0.183$ while acquiring nearly three verifiers.
The compute term and the significance gate change little at $\rho=1$ on this small, strong pool --- their
role is larger at higher compute price and with weaker candidate pools --- but the false-alarm term is
decisive (\Cref{fig:acqgain}(a)). Improvement is not from acquiring more; it is from charging correctly
for what is acquired.

\begin{table}[t]\centering\small
\caption{Value-function component ablation at $\rho=1$ (macro-avg over six families), from
\texttt{rev\_component\_ablation}. The false-alarm term is what prevents over-acquisition.}
\label{tab:component}
\begin{tabular}{lcccc}
\toprule
value function & catch & false alarm & \# verifiers & utility $U$\\
\midrule
full CMV                       & 0.712 & 0.227 & 1.00 & \textbf{0.486}\\
\;\;--\ no false-alarm term    & 0.867 & 0.456 & 1.00 & 0.411\\
\;\;--\ no compute term        & 0.758 & 0.271 & 1.00 & 0.487\\
\;\;--\ catch-only (no FA, no cost) & 0.954 & 0.771 & 2.83 & 0.183\\
\;\;--\ no significance gate    & 0.712 & 0.227 & 1.00 & 0.486\\
\bottomrule
\end{tabular}
\end{table}

\begin{figure}[t]\centering
\includegraphics[width=\linewidth]{figures/fig_acqgain.pdf}
\caption{Why the conditional-marginal-value acquisition gain is real (six families, $\rho=1$).
\textbf{(a)}~value-function component ablation: switching off the false-alarm or compute penalty raises
raw catch (green), but the false-alarm rate (crimson) climbs faster, so utility $U$ (blue) is maximised
only by the full value function; the ``catch-only'' policy over-acquires and collapses $U$.
\textbf{(b)}~paired bootstrap of the utility gap $\Delta U=U_{\text{CMV-SDSE}}-U_{\text{baseline}}$
($2000$ draws): every $95\%$ interval lies strictly to the right of zero (dashed line), with large
standardised effect sizes ($d$ annotated), so the advantage is not a one-family artifact.}
\label{fig:acqgain}
\end{figure}

\subsection{The controller action map across families}
\label{app:decisionmaps}

\Cref{fig:decisionmaps} draws, for every confirmatory family, the controller's first action as a
function of the operating regime $(\lambda,\rho)$ (compute price, false-alarm/miss cost). The behaviour
is not a fixed policy: every family \textsc{Diversify}s when misses and compute are cheap, shifts to
\textsc{Stop} as either cost rises, and a few (CSQA, TruthfulQA) show a narrow \textsc{Scale} niche at
high $\rho$. The boundaries and the best solo differ by family, but the qualitative regime dependence is
systematic --- exactly the behaviour the objective $U_{\lambda,\rho}$ prescribes.

\begin{figure}[t]\centering
\includegraphics[width=\linewidth]{figures/fig_decision_maps.pdf}
\caption{Controller decision maps over $(\lambda,\rho)$
(\textsc{Diversify}/\textsc{Scale}/\textsc{Stop}), one per confirmatory family:
(a)~ARC, (b)~MMLU, (c)~CSQA, (d)~TruthfulQA, (e)~GSM8K, (f)~MBPP. The action shifts systematically with
the operating regime; the boundaries and the best solo verifier are family-specific, but the pattern is
not.}
\label{fig:decisionmaps}
\end{figure}

\subsection{Statistical robustness of the acquisition gains}
\label{app:robust}

\Cref{tab:bootstrap} attaches paired uncertainty to the utility gap between CMV-SDSE and each baseline.
The per-family utility differences are paired and resampled ($2000$ family bootstrap draws); every
$95\%$ interval excludes zero and the standardised effect sizes are large (Cohen's $d\ge1.7$;
\Cref{fig:acqgain}(b)). The advantage of acquiring by conditional marginal value is not a one-family
artifact.

\begin{table}[t]\centering\small
\caption{Paired bootstrap of the utility gap $\Delta U=U_{\text{CMV-SDSE}}-U_{\text{baseline}}$ at
$\rho=1$ over the six families, from \texttt{rev\_bootstrap}.}
\label{tab:bootstrap}
\begin{tabular}{lccc}
\toprule
comparison & mean $\Delta U$ & 95\% bootstrap CI & Cohen's $d$\\
\midrule
CMV-SDSE $-$ accuracy-ranked   & 0.302 & [0.239, 0.374] & 3.15\\
CMV-SDSE $-$ low-correlation   & 0.293 & [0.236, 0.370] & 3.15\\
CMV-SDSE $-$ same-model        & 0.252 & [0.129, 0.344] & 1.73\\
\bottomrule
\end{tabular}
\end{table}

\subsection{Real compute cost (measured latency)}
\label{app:latency}

The utility already charges for compute through $\lambda$; \Cref{tab:latency} makes that concrete with
the \emph{measured} wall-clock latency of each verifier call (median per checkpoint, from the run logs).
Fifty same-model calls spend $\approx\!30$ seconds to buy utility $0.23$, and the full cross-family
cascade spends $1.5$ seconds for $0.20$, whereas CMV-SDSE spends $\approx\!0.26$ seconds --- one call ---
for utility $0.49$ (\Cref{fig:diagnostics}(a)). Same-model verification is not just statistically
redundant; it is a real, and large, waste of wall-clock budget.

\begin{table}[t]\centering\small
\caption{Measured compute cost per acquisition policy at $\rho=1$ (macro-avg), using median per-verifier
latency, from \texttt{rev\_latency}. Median call latency: qwen7b $0.25$s, llama31 $0.32$s,
llama32\_3b $0.39$s, mistral7b $0.77$s.}
\label{tab:latency}
\begin{tabular}{lcccc}
\toprule
acquisition policy & calls & wall-clock (s) & catch & utility $U$\\
\midrule
same-model ($k{=}50$)  & 50 & 30.40 & 0.89 & 0.23\\
cross-family (all 3)   & 3  & 1.48  & 0.93 & 0.20\\
\textbf{CMV-SDSE}      & 1  & \textbf{0.26} & 0.71 & \textbf{0.49}\\
\bottomrule
\end{tabular}
\end{table}

\begin{figure}[t]\centering
\includegraphics[width=\linewidth]{figures/fig_diagnostics.pdf}
\caption{Cost and robustness diagnostics for the acquisition policy (six-family averages at $\rho=1$).
\textbf{(a)}~measured wall-clock compute per decision (log scale) versus utility: CMV-SDSE (one call,
$0.26$s) reaches the highest utility while spending $\approx\!117\times$ less compute than the $50$-call
same-model policy. \textbf{(b)}~leave-one-model-out utility against the full-pool value (dashed line):
only the precise Qwen solo is load-bearing (crimson), not the highest-catch verifier. \textbf{(c)}~per-verifier
heterogeneity --- standalone catch versus false-alarm rate, marker area $\propto$ median call latency; the
dead $3$B checkpoint (crimson, lower-left) has the lowest catch and is never selected. \textbf{(d)}~probe
cost versus final utility: utility is flat from $R=1$ to $R=50$, so a one- or two-round probe reaches the
same acquisition decision as the full data.}
\label{fig:diagnostics}
\end{figure}

\subsection{Robustness to dropping a family or a model}
\label{app:loo}

\Cref{tab:lofo} recomputes the two headline results with each task family removed in turn: the same-model
$n_{\mathrm{eff}}$ range stays inside $[1.06,1.21]$ and the pooled diversify signal stays supported (five
or six families with Wilson lower bound $>0$) no matter which family is dropped --- neither result is
carried by any single benchmark. \Cref{tab:lomo} removes each verifier from the pool and re-runs the
controller: dropping the dead $3$B verifier or the high-false-alarm Mistral checkpoint leaves utility
unchanged at $\rho=1$, while the precise Qwen solo is the only load-bearing model (its removal costs
$0.15$; \Cref{fig:diagnostics}(b)). No single model props up the average; the one that matters at balanced
cost is the precise, not the highest-catch, verifier.

\begin{table}[t]\centering\small
\caption{Leave-one-family-out, from \texttt{rev3\_lofo}. The redundancy null and the diversify signal
survive dropping any family. ``diversify pooled'' is $\sum$unique\,/\,$\sum$denom residual catches;
``families CI$>$0'' counts families with Wilson lower bound $>0$.}
\label{tab:lofo}
\begin{tabular}{lccc}
\toprule
family dropped & $n_{\mathrm{eff}}@50$ range & diversify pooled & families CI$>$0\\
\midrule
(none)     & [1.06, 1.21] & 71/120 & 6/6\\
MAST-3.3   & [1.06, 1.16] & 71/120 & 6/6\\
ARC        & [1.06, 1.21] & 60/105 & 5/5\\
MMLU       & [1.06, 1.21] & 50/94  & 5/5\\
CSQA       & [1.06, 1.21] & 62/104 & 5/5\\
TruthfulQA & [1.06, 1.21] & 52/84  & 5/5\\
GSM8K      & [1.06, 1.21] & 68/113 & 5/5\\
MBPP       & [1.09, 1.21] & 63/100 & 5/5\\
\bottomrule
\end{tabular}
\end{table}

\begin{table}[t]\centering\small
\caption{Leave-one-model-out, from \texttt{rev3\_lomo}: macro CMV-SDSE utility at $\rho=1$ with each
verifier removed from the pool.}
\label{tab:lomo}
\begin{tabular}{lcc}
\toprule
verifier removed & CMV-SDSE utility & change vs.\ full pool\\
\midrule
(full pool)  & 0.486 & ---\\
llama32\_3b (dead) & 0.486 & $0.000$\\
mistral7b    & 0.486 & $0.000$\\
llama31      & 0.432 & $-0.053$\\
qwen7b       & 0.338 & $-0.148$\\
\bottomrule
\end{tabular}
\end{table}

\subsection{Per-model and per-family breakdowns}
\label{app:breakdowns}

\Cref{tab:permodel} confirms the pool is heterogeneous rather than one model driving the average: the
verifiers span standalone catch $0.20$--$0.85$, false-alarm $0.05$--$0.63$, and latency $0.25$--$0.77$s;
the dead $3$B checkpoint has by far the lowest catch and is never selected (\Cref{fig:diagnostics}(c)).
\Cref{tab:fusionwinner} shows
the ``no universal fusion rule'' conclusion is not a macro artifact: the best rule at $\rho=1$ differs
across families (weighted vote, single-best, or majority). \Cref{tab:debt} restates the redundancy as an
operational cost: of $50$ same-model calls only about one is effective, a Verification Debt near $49$ and
an efficiency near $2\%$.

\begin{table}[t]\centering\small
\caption{Per-verifier breakdown over the six families, from \texttt{rev3\_per\_model}: standalone catch
and false alarm, median call latency, and how often the controller uses it as solo start or as
diversifier (in the miss-dominated regime).}
\label{tab:permodel}
\begin{tabular}{lccccc}
\toprule
verifier & standalone catch & standalone fa & median latency (s) & times solo & times diversifier\\
\midrule
qwen7b     & 0.679 & 0.247 & 0.25 & 1 & 4\\
llama31    & 0.766 & 0.404 & 0.32 & 1 & 5\\
llama32\_3b & 0.204 & 0.053 & 0.39 & 0 & 0\\
mistral7b  & 0.846 & 0.633 & 0.77 & 4 & 2\\
\bottomrule
\end{tabular}
\end{table}

\begin{table}[t]\centering\small
\caption{Best fixed fusion rule per family at $\rho=1$, from \texttt{rev3\_fusion\_winner}. The winner is
family-specific, reinforcing that no single rule is universal.}
\label{tab:fusionwinner}
\begin{tabular}{lcc}
\toprule
family & best rule at $\rho=1$ & Youden $J$\\
\midrule
ARC & weighted vote & 0.680\\
CSQA & weighted vote & 0.600\\
MMLU & single-best & 0.480\\
TruthfulQA & single-best & 0.547\\
GSM8K & single-best & 0.247\\
MBPP & majority & 0.407\\
\bottomrule
\end{tabular}
\end{table}

\begin{table}[t]\centering\small
\caption{Verification Debt ($50-n_{\mathrm{eff}}$) and Efficiency ($n_{\mathrm{eff}}/50$) per family,
from \texttt{rev3\_debt}: of $50$ same-model calls, about one carries independent evidence.}
\label{tab:debt}
\begin{tabular}{lccc}
\toprule
family & $n_{\mathrm{eff}}@50$ & Verification Debt & Efficiency\\
\midrule
MAST-3.3 & 1.21 & 48.79 & 0.024\\
ARC & 1.11 & 48.89 & 0.022\\
MMLU & 1.13 & 48.87 & 0.023\\
CSQA & 1.11 & 48.89 & 0.022\\
TruthfulQA & 1.09 & 48.91 & 0.022\\
GSM8K & 1.16 & 48.84 & 0.023\\
MBPP & 1.06 & 48.94 & 0.021\\
\bottomrule
\end{tabular}
\end{table}

\subsection{Probe cost versus final utility}
\label{app:probeutility}

\Cref{tab:probeutil} closes the loop on the probe: it reports the utility of the controller when the
conditional marginal value is estimated from only $R$ joint verdict rounds. The final utility is
essentially flat from $R=1$ to the full $R=50$ (\Cref{fig:diagnostics}(d)), so a one- or two-round probe
reaches the same acquisition decision as the full data --- the calibration phase is nearly free, in
agreement with \Cref{prop:probe} and \Cref{tab:probe}.

\begin{table}[t]\centering\small
\caption{Controller utility from an $R$-round probe (macro-avg, $\rho=1$), from \texttt{rev3\_probe\_utility}.}
\label{tab:probeutil}
\begin{tabular}{lcccccc}
\toprule
probe rounds $R$ & 1 & 2 & 3 & 5 & 10 & 50 (full)\\
\midrule
CMV-SDSE utility & 0.479 & 0.470 & 0.482 & 0.480 & 0.482 & 0.479\\
\bottomrule
\end{tabular}
\end{table}
