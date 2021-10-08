package group3;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Random;
import java.util.Set;
import java.util.concurrent.ThreadLocalRandom;
import java.util.stream.Collectors;

import genius.core.Bid;
import genius.core.bidding.BidDetails;
import genius.core.boaframework.BOAparameter;
import genius.core.boaframework.NegotiationSession;
import genius.core.boaframework.NoModel;
import genius.core.boaframework.OMStrategy;
import genius.core.boaframework.OfferingStrategy;
import genius.core.boaframework.OpponentModel;
import genius.core.boaframework.SortedOutcomeSpace;
import genius.core.issue.Issue;
import genius.core.issue.IssueDiscrete;
import genius.core.issue.IssueInteger;
import genius.core.issue.IssueReal;
import genius.core.issue.Value;
import genius.core.issue.ValueInteger;
import genius.core.issue.ValueReal;
import genius.core.misc.Range;


public class Group3_BS extends OfferingStrategy {

	/**
	 * k in [0, 1]. For k = 0 the agent starts with a bid of maximum utility
	 */
	private double k;
	/** Maximum target utility */
	private double Pmax;
	/** Minimum target utility */
	private double Pmin;
	/** Concession factor */
	private double e;
	/** Outcome space */
	private SortedOutcomeSpace outcomespace;
	
	// Switch to bram after time Threshold (aanpassen)
	private double toBramTimeThreshold = 0.5;
	// should be final after init
	protected Random rand;

	/**
	 * Method which initializes the agent by setting all parameters. The
	 * parameter "e" is the only parameter which is required.
	 */
	@Override
	public void init(NegotiationSession negoSession, OpponentModel model, OMStrategy oms,
			Map<String, Double> parameters) throws Exception {
		super.init(negoSession, parameters);
		if (parameters.get("e") != null) {
			this.rand = new Random();
			this.negotiationSession = negoSession;

			outcomespace = new SortedOutcomeSpace(negotiationSession.getUtilitySpace());
			negotiationSession.setOutcomeSpace(outcomespace);

			this.e = parameters.get("e");

			if (parameters.get("k") != null)
				this.k = parameters.get("k");
			else
				this.k = 0;

			if (parameters.get("min") != null)
				this.Pmin = parameters.get("min");
			else
				this.Pmin = negoSession.getMinBidinDomain().getMyUndiscountedUtil();

			if (parameters.get("max") != null) {
				Pmax = parameters.get("max");
			} else {
				BidDetails maxBid = negoSession.getMaxBidinDomain();
				Pmax = maxBid.getMyUndiscountedUtil();
			}

			this.opponentModel = model;
			
			this.omStrategy = oms;
		} else {
			throw new Exception("Constant \"e\" for the concession speed was not set.");
		}
	}

	@Override
	public BidDetails determineOpeningBid() {
		return determineNextBid();
	}

	/**
	 * Tat for Tit bidding strategy
	 */
	@Override
	public BidDetails determineNextBid() {
		double time = this.negotiationSession.getTime();
		double maximumUtility = negotiationSession.getMaxBidinDomain().getMyUndiscountedUtil();	
		
	
		// Stel ons beste bid als eerste voor
		if (this.negotiationSession.getOpponentBidHistory().getHistory().isEmpty()) {
			return this.negotiationSession.getMaxBidinDomain();
		}
		
			
		// Perform tat for tit untill bramThreshold is reached 
		if (time < this.toBramTimeThreshold) {
			// Get target utility (tat for tit)
			List<BidDetails> opponentOfferHistory = new ArrayList<BidDetails>(this.negotiationSession.getOpponentBidHistory().getHistory());
			BidDetails lastOffer = opponentOfferHistory.get(opponentOfferHistory.size() - 1);
			double targetUtil = 1 - lastOffer.getMyUndiscountedUtil();
			
			// Find bid closest to target utility
			return this.outcomespace.getBidNearUtility(targetUtil);
		}
		
		// Bram utility threshold decay
		double utilityThreshold = maximumUtility * 0.93;
		if (time >= (1 - this.toBramTimeThreshold) * 0.3 + this.toBramTimeThreshold)
			utilityThreshold = maximumUtility * 0.85;
		if (time >= (1 - this.toBramTimeThreshold) * 0.8 + this.toBramTimeThreshold)
			utilityThreshold = maximumUtility * 0.7;
		if (time >= (1 - this.toBramTimeThreshold) * 0.94 + this.toBramTimeThreshold)
			utilityThreshold = maximumUtility * 0.2;
		
		// Get opponent outcomespace 
		Range utilityRange = new Range(utilityThreshold, 1);		
		SortedOutcomeSpace opponentOutcomespace = new SortedOutcomeSpace(this.opponentModel.getOpponentUtilitySpace());
		
		// Get all bids above threshold for both agent and opponent
		List<Bid> opponentPossibleBids = opponentOutcomespace.getBidsinRange(utilityRange)
				.stream().filter(elt -> elt != null)
				.map(elt -> elt.getBid())
				.collect(Collectors.toList());
		List<Bid> possibleBids = this.outcomespace.getBidsinRange(utilityRange)
				.stream().filter(elt -> elt != null)
				.map(elt -> elt.getBid())
				.collect(Collectors.toList());
		
		// Get bids with utility above threshold for both agent and opponent
		opponentPossibleBids.retainAll(possibleBids);

		// Check if win-win bid is possible
		if(opponentPossibleBids.size() > 0) {
			Integer randomIndex = ThreadLocalRandom.current().nextInt(opponentPossibleBids.size());			
			
			// Take random win-win bid
			Bid randomBid = opponentPossibleBids.get(randomIndex);
			double bidUtil = this.negotiationSession.getUtilitySpace().getUtility(randomBid);			
			return new BidDetails(randomBid, bidUtil);
		}
		
		// Else return bid with utility above threshold for agent
		Integer randomIndex = ThreadLocalRandom.current().nextInt(possibleBids.size());			
		
		// Take random bid above threshold
		Bid randomBid = possibleBids.get(randomIndex);
		double bidUtil = this.negotiationSession.getUtilitySpace().getUtility(randomBid);
		return new BidDetails(randomBid, bidUtil);
	}
	
	protected Bid generateRandomBid() {
		try {
			HashMap<Integer, Value> values = new HashMap<Integer, Value>();

			// For each issue, put a random value
			for (Issue currentIssue : negotiationSession.getUtilitySpace().getDomain().getIssues()) {
				values.put(currentIssue.getNumber(), getRandomValue(currentIssue));
			}

			// return the generated bid
			return new Bid(negotiationSession.getUtilitySpace().getDomain(), values);

		} catch (Exception e) {

			// return empty bid if an error occurred
			return new Bid(negotiationSession.getUtilitySpace().getDomain());
		}
	}

	/**
	 * Gets a random value for the given issue.
	 *
	 * @param currentIssue
	 *            The issue to generate a random value for
	 * @return The random value generated for the issue
	 * @throws Exception
	 *             if the issues type is not Discrete, Real or Integer.
	 */
	protected Value getRandomValue(Issue currentIssue) throws Exception {

		Value currentValue;
		int index;

		switch (currentIssue.getType()) {
		case DISCRETE:
			IssueDiscrete discreteIssue = (IssueDiscrete) currentIssue;
			index = (rand.nextInt(discreteIssue.getNumberOfValues()));
			currentValue = discreteIssue.getValue(index);
			break;
		case REAL:
			IssueReal realIss = (IssueReal) currentIssue;
			index = rand.nextInt(realIss.getNumberOfDiscretizationSteps()); // check
																			// this!
			currentValue = new ValueReal(
					realIss.getLowerBound() + (((realIss.getUpperBound() - realIss.getLowerBound()))
							/ (realIss.getNumberOfDiscretizationSteps())) * index);
			break;
		case INTEGER:
			IssueInteger integerIssue = (IssueInteger) currentIssue;
			index = rand.nextInt(integerIssue.getUpperBound() - integerIssue.getLowerBound() + 1);
			currentValue = new ValueInteger(integerIssue.getLowerBound() + index);
			break;
		default:
			throw new Exception("issue type " + currentIssue.getType() + " not supported");
		}

		return currentValue;
	}

	
	public double f(double t) {
		if (e == 0)
			return k;
		double ft = k + (1 - k) * Math.pow(t, 1.0 / e);
		return ft;
	}

	  
	  @param t
	  @return //double
	public double p(double t) {
		return Pmin + (Pmax - Pmin) * (1 - f(t));
	}

	public NegotiationSession getNegotiationSession() {
		return negotiationSession;
	}

	@Override
	public Set<BOAparameter> getParameterSpec() {
		Set<BOAparameter> set = new HashSet<BOAparameter>();
		set.add(new BOAparameter("e", 1.0, "Concession rate"));
		set.add(new BOAparameter("k", 0.0, "Offset"));
		set.add(new BOAparameter("min", 0.0, "Minimum utility"));
		set.add(new BOAparameter("max", 0.99, "Maximum utility"));

		return set;
	}

	@Override
	public String getName() {
		return "TimeDependent Offering example";
	}
}