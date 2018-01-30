



;;; The template definition for the sports. 

(deftemplate sport "Template holding the characteristics of a sport."
	

	;;; List of attributes for this template.
	(multislot name (type STRING)(default ?NONE))
	(multislot kind (type STRING)(default ?NONE))
	(multislot size (type NUMBER)(default ?NONE))
	

	(multislot budget (type NUMBER)(default ?NONE)
		;;;(allowed-symbols 10 25 30 60 70 50 80 40 65 75 100 3000 4000 5000 1200 12000)
	)
	(multislot water (type SYMBOL) (default false) (allowed-symbols false true))
	(multislot time (type SYMBOL) (default ?DERIVE)
		;;;(allowed-symbols short-time medium-time long-time)
	)
	
	
	(multislot requirement (type SYMBOL) (default ?NONE)
	       ;;;(allowed-symbols soapwater tube pressure-gun board die goal-post ball racket shuttlecock quanco)
	)
	
	
	(multislot location (type SYMBOL) (default ?DERIVE)
		;;;(allowed-symbols home garden central-asia ground court field arena Australia central-asia middle-east himanchal dubai goa)
	)
)

;;;set of facts for each sport being defined.

(deffacts sport-database 
	
	(sport (name "Air-bubble")   					  
			(kind "indoor")   
			(size 2)                     
			(budget 10 30 60 )	
			(water true)				 
			(time long-time)
			(requirement soapwater tube)		
			(location home)
			
	)
           				
	(sport (name "water-guns")                   			            
			(kind "indoor")	
			(size 3)
			(budget 25)  	
			(water true)				
			(time shorttime)
			(requirement pressure-gun )
			(location patio)
	)
                				
	(sport (name "Ludo")                   			            
			(kind "indoor")
			(size 4)	
			(budget 10 70)
			(water false)  					
			(time short-time mediumtime)
			(requirement board die)
			(location garden central-asia)
			
	)
                 				
	(sport (name "Poker")                   			            
			(kind "indoor")	
			(size 5)
			(budget 70 10 25)  
			(water false)					
			(time long-time)
			(requirement board)
			(location casino)
	)


	(sport (name "Football")     
			(kind "Outdoor")   
			(size 7)        
			(budget 50 80)
			(water false)	
			(time long-time medium-time)	      
			(requirement goal-post ball)		
			(location ground)
			
	)		
	
	(sport (name "Badmington")	
			(kind "outdoor")
			(size 8)	
			(budget 40 75)
			(water false)			
			(time short-time)
			(requirement racket shuttlecock)			
			(location court)
			
	)	
	
	(sport (name "Rugby")	
			(kind "outdoor")
			(size 9)	
			(budget 65 50)			
			(water false)	
			(time mediumtime)
			(requirement quanco)	
			(location field)
			
	)

	(sport (name "Basketball")	
			(kind "outdoor")
			(size 8)	
			(budget 80)
			(water false)			
			(time long-time)
			(requirement ball)		
			(location arena)
			
	)
             
	(sport (name "Paragliding")		
			(kind "adventure")
			(size 12)	
			(budget 5000 3000)
			(water false)	
			(time medium-time)	
			(requirement life-jacket gear)		
			(location Australia cenral-asia)
			
	)		

	(sport (name "Rafting")
			(kind "adventure")
			(size 14)	
			(budget 5000)	
			(water false)
			(time long-time)	
			(requirement raft life-jacket)	
			(location middle-east)
			
	)

	(sport (name "Bungee-jumping")
			(kind "adventure")
			(size 11)	
			(budget 1200 3000)
			(water true)
			(time shorttime)
			(requirement life-jacket rope)
			(location himanchal)
			
	)	

	(sport (name "Parasailing")
			(kind "adventure")
			(size 13)	
			(budget 1200 4000)
			(water true)
			(time medium-time)
			(requirement boat paraschute)		
			(location dubai goa)
			
	)
)

;;; A variable that holds the number of avaliable animals.
(defglobal ?*counter* = 12)


;;; The variable counter declared above is being modified each time we exclude a sport from
;;; the possible choices.
(deffunction minusOne ()
	(bind ?*counter* (- ?*counter* 1))
)


;;; This function is used for each and every question asked to the user.

;;; If the user enters a non-acceptable value, the program will keep asking the same question again until the answer is valid.
(deffunction ask-question (?qBEG ?qMID ?qEND $?allowed-values)
	(printout t ?qBEG ?qMID ?qEND)
	(bind ?answer (read))
	(if (lexemep ?answer)
		then (bind ?answer (lowcase ?answer))
	)
	(while (not (member ?answer ?allowed-values)) do
		(printout t ?qBEG ?qMID ?qEND)
		(bind ?answer (read))
		(if (lexemep ?answer)
			then (bind ?answer (lowcase ?answer)))
	)
?answer)


;;; The first main question made to the user. We ask for the number of people( group-size) that the user wants the game to have.

(defrule mainQuestion-Size
	?x <- (initial-fact)
	=>
	(retract ?x)
	(bind ?size (ask-question "### Is the sport indoor ( <6 ) , outdoor ( 6-10 ) or adventure ( >10 )? (indoor,outdoor,adventure) ### " "" "" indoor outdoor adventure))
	(assert (theScale ?size))	
)


;;; Given that the fact (theScale ?size) is presnt as an acceptable value, this rule gets called.


(defrule filterBy-Size
	(theScale ?s)
	?spo <- (sport (size ?size))
	=>
	(if (eq ?s indoor)
		then (if (> ?size 6) then (retract ?spo) (minusOne))
	else (if (eq ?s outdoor)
		    then (if (or (<= ?size 6) (>= ?size 10)) then (retract ?spo) (minusOne))
		 )
	else (if (eq ?s adventure)
		    then (if (< ?size 10) then (retract ?spo) (minusOne))
		 )
	)
)


;;; The second main question is about the budget of the sport.

(defrule mainQuestion-Budget
	(theScale ?s)								
	=>
	(bind ?budget (ask-question "### What is the budget? (1200,25,30,60,70,65,75,80,100,50,40,20,4000,3000,10,5000) ### " "" "" 1200 25 30 60 70 65 75 80 100 50 40 20 4000 3000 10 5000))
	(assert (theBudget ?budget))    
)	

;;;Now we filter the sports by their indicated budget, giving the user required response or asking more questions if needed.
(defrule filterBy-Budget
	(theBudget ?c)
	?spo <- (sport (budget $?budget))
	=>
	(if (not (member$ ?c $?budget))
		then (retract ?spo) (minusOne)
	)
)


;;; Here we decide whether this is answer user wants or do we need to check out more facts. Counter checks if this is the last entry, if so this is the ;;;response.

(defrule postFilteringEvaluation
    ?scale <- (theScale ?s)
	?budget <- (theBudget ?c)
	=>
	(retract ?scale ?budget)
	(if (eq ?*counter* 1)
		then (assert (found true))
	else (if (eq ?*counter* 0)
			then (assert (found false))
		 ) 
	else (if (> ?*counter* 1)
			then (assert (needMoreFacts ?s ?c))
		 ) 
	)	
)	


;;;This is the rule being referred above and called as and when needed.

(defrule needMoreFacts
	?q <-(needMoreFacts ?s ?c)
	=>
	(retract ?q)
	(if (and (eq ?s indoor) (eq ?c 10))
		then (assert (ask water true))
	)
	(if (and (eq ?s indoor) (eq ?c 70))
		then (assert (ask time short-time))
	)
	(if (and (eq ?s indoor) (eq ?c 25))               
		then (assert (ask requirement pressure-gun))
	)
	(if (and (eq ?s outdoor) (eq ?c 50))
		then (assert (ask time long-time))
	)
	(if (and (eq ?s outdoor) (eq ?c 80))
		then (assert (ask time medium-time))
	)
	(if (and (eq ?s adventure) (eq ?c 5000))
		then (assert (ask requirement gear))
	)
	(if (and (eq ?s adventure) (eq ?c 1200))
		then (assert (ask requirement paraschute))
	)
	(if (and (eq ?s adventure) (eq ?c 3000))
		then (assert (ask location Australia))
	)
)


;;; Another rule which accpets a boolean value as entry.

(defrule askwater				
	?q <-(ask water true)
	?spo1 <- (sport (water true))
	?spo2 <- (sport (water false))
	?spo3 <- (sport (water false))
	(test (neq ?spo2 ?spo3))
	=>
	(retract ?q)
	(bind ?a (ask-question "### Does the sport involve " water "? (yes/no) ### " yes no))
	(if (eq ?a yes)
		then (retract ?spo2) (minusOne) (retract ?spo3) (minusOne)
		else (retract ?spo1) (minusOne) (assert (ask location Middle-east))
	)
	(if (eq ?*counter* 1)
		then (assert (found true))
	)
)

;;; This rule follows the above rule in the manner that this rule also accepts boolean values.

(defrule asktime
	?q <-(ask time ?ans)			
	?spo1 <- (sport (time $?content1))
	(test (member$ ?ans $?content1))
	?spo2 <- (sport (time $?content2))
	(test (neq ?spo1 ?spo2))
	=>
	(retract ?q)
	(bind ?a (ask-question "### Does the sport take a " ?ans " to play? (yes/no) ### " yes no))
	(if (eq ?a yes)
		then (retract ?spo2) (minusOne)
		else (retract ?spo1) (minusOne)
	)
	(if (eq ?*counter* 1)
		then (assert (found true))
	)
)



;;; This rule again accepts boolean values and makes reuqired decision.


(defrule askrequirement
	?q <-(ask requirement ?ans)
	?spo1 <- (sport (requirement $?content1))
	(test (member$ ?ans $?content1))
	?spo2 <- (sport (requirement $?content2))
	(test (neq ?spo1 ?spo2))
	=>
	(retract ?q)
	(bind ?a (ask-question "### Does the sport have a requirement that contains " ?ans "? (yes/no) ### " yes no))
	(if (eq ?a yes)
		then (retract ?spo2) (minusOne)
		else (retract ?spo1) (minusOne)
	)
	(if (eq ?*counter* 1)
		then (assert (found true))
	)
)

;;; This rule gets triggered if the previous rule was reached by the set of the questioning as directed.
(defrule asklocation								
	?q <-(ask location ?ans)
	?spo1 <- (sport (location $?content1))
	(test (member$ ?ans $?content1))
	?spo2 <- (sport (location $?content2))
	(test (neq ?spo1 ?spo2))
	=>
	(retract ?q)
	(bind ?a (ask-question "### Is the sport located in " Australia"? (yes/no) ### " yes no))
	(if (eq ?a yes)
		then (retract ?spo2) (minusOne)
		else (retract ?spo1) (minusOne)
	)
	(if (eq ?*counter* 1)
		then (assert (found true))
	)
)

;;; If the fact is present, we present the user information of this sport.
(defrule matchFound
	?f <- (found true)
	?spo <- (sport (name ?n) (kind ?k))
	=>
	(retract ?f ?spo)
	(printout t "*********************" crlf)
	(printout t "* Sport found!" crlf)
	(printout t "* Name: " ?n crlf)
	(printout t "* Kind: " ?k crlf)
	(printout t "*********************" crlf)
)

;;; If no choices match the rules we have laid out this si the message user receives.
(defrule matchNotFound
	?f <- (found false)
	=>
	(retract ?f)
	(printout t "*********************" crlf)
	(printout t "* No sports match your criteria!" crlf)
	(printout t "*********************" crlf)
)