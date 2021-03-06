import scala.io.Source
import java.io._
import java.lang.String
import breeze.linalg._
import breeze.optimize._
import breeze.stats.distributions._
import breeze.numerics._
import util.control.Breaks._
import scala.collection.mutable.ListBuffer

object CRF {
/*Saves everything in a list buffer*/
val buf : ListBuffer[(String,String)] = ListBuffer()
val fileiter : Iterator[String] = scala.io.Source.fromResource("train.txt").getLines()
	while(fileiter.hasNext) {
        var line = fileiter.next()
        if (line.length() > 0) {
         	var splitLine = (line).split(" ")
            splitLine(1) match {
                case "NN" => splitLine(1) = "NOUN"
                case "NNS" => splitLine(1) = "NOUN"
                case "NNP" => splitLine(1) = "NOUN"
                case "NNPS" => splitLine(1) = "NOUN"
                case "VB" => splitLine(1) = "VERB"
                case "VBD" => splitLine(1) = "VERB"
                case "VBG" => splitLine(1) = "VERB"
                case "VBN" => splitLine(1) = "VERB"
                case "VBP" => splitLine(1) = "VERB"
                case "VBZ" => splitLine(1) = "VERB"
                case _ => splitLine(1) = "OTHER"
            }
            buf += ((splitLine(0),splitLine(1)))
        }
  	}
  //println(buf(0)._1 + buf(1)._2)

/*Properties of model*/
val k = 5 //5-gram model
val m = 2 //2 features 
type label = String
type token = String
val labels : List[label] = List("NOUN", "VERB", "OTHER")
var currpos = 1 //currentposition in the list buffer.



/*Feature functions*/
//VERB
def f1(yi: label, yimin1: label, tokens: Array[token], i: Int): Double = {
      //  println("f1")
        if (tokens(i).endsWith("ing") && yi == "VERB") {
            return 1.0
        }
        else {
            return 0.0
        }
    }
 //OTHER
def f2(yi: label, yimin1: label, tokens: Array[token], i: Int): Double = {
  //  println("f2")
	if (tokens(i).endsWith("ly") && yi == "OTHER") {
            return 1.0
        }
        else {
            return 0.0
        }
 }

 def f3(yi: label, yimin1: label, tokens: Array[token], i: Int): Double = {
    //println("f3")
    if (i == 0) {return 0.0}
	else if (((tokens(i-1) == "the") || (tokens(i-1) == "a"))&& yi == "NOUN") {
            return 1.0
        }
        else {
            return 0.0
        }
 }
 def f4(yi: label, yimin1: label, tokens: Array[token], i: Int): Double = {
  //  println("f4")
	if (tokens(i).endsWith("ive") && yi == "OTHER") {
            return 1.0
        }
        else {
            return 0.0
        }
 }
 def f5(yi: label, yimin1: label, tokens: Array[token], i: Int): Double = {
  //  println("f5")
	if (yimin1 == "OTHER" && yi == "NOUN") {
            return 1.0
        }
        else {
            return 0.0
        }
 }
 def f6(yi: label, yimin1: label, tokens: Array[token], i: Int): Double = {
  //  println("f6")
	if (yimin1 == "VERB" && ((yi == "NOUN") || (yi == "OTHER"))) {
            return 1.0
        }
        else {
            return 0.0
        }
 }
 def f7(yi: label, yimin1: label, tokens: Array[token], i: Int): Double = {
  //  println("f7")
	if ((tokens(i) == ".") || (tokens(i) == ",") && yi == "OTHER") {
            return 1.0
        }
        else {
            return 0.0
        }
 }



//The Model
def weightedfeatures(y: Array[label], x: Array[token], theta:DenseVector[Double]):Double = {
   	//TODO we need this to be 1 when single sentence, 0 when long sentences
   //	println("weightedfeatures")
   //	println(x(0))
   //TODO this really shud be 1 page 288
    var features = for(i <- 1 until k) yield {
        exp(theta(0)*f1(y(i),y(i-1),x,i) + theta(1)*f2(y(i),y(i-1),x,i)
        + theta(2)*f3(y(i),y(i-1),x,i) + theta(3)*f4(y(i),y(i-1),x,i)
        + theta(4)*f5(y(i),y(i-1),x,i) + theta(5)*f6(y(i),y(i-1),x,i)
        + theta(6)*f7(y(i),y(i-1),x,i)
    	)
    }
    //edwin chen says to add them lol 
  //  println("pos ion")
    var toreturn = features.reduceLeft(_*_)
    //println("WEIGHTEDFEAT" + toreturn)
    return toreturn
}

def normalize(x: Array[token], theta: DenseVector[Double]):Double = {
   // println("normalize")
    var features = for(y1<-labels;y2<-labels;y3<-labels;y4<-labels;y5<-labels) yield {
        weightedfeatures(Array(y1,y2,y3,y4,y5), x, theta)
    }
    features.reduceLeft(_+_)
}

def probability(y: Array[label], x: Array[token], theta: DenseVector[Double]):Double = {
    var toreturn = weightedfeatures(y,x,theta)/normalize(x,theta)
   // println("PRO" + toreturn)
    return toreturn
}




//Feature Engineering
def neglog_likelihood(theta: DenseVector[Double]): Double = {
    println("loglike theta: " + theta)
	if (currpos == buf.length) { break }
    println("array: " +   buf(currpos)._1 + buf(currpos + 1)._1 + buf(currpos + 2)._1 + buf(currpos + 3)._1 + buf(currpos + 4)._1)
	var features = for(j <- 0 until k) yield {
		//println(j + ": j")
		//println(currpos + ": currpos")
		//println("array: " +   buf(currpos)._1 + buf(currpos + 1)._1 + buf(currpos + 2)._1 + buf(currpos + 3)._1 + buf(currpos + 4)._1)
	    theta(0)*f1(buf(currpos + j)._2,buf(currpos + (j-1))._2,Array[String](buf(currpos)._1,
	                                                      buf(currpos + 1)._1,
	                                                      buf(currpos + 2)._1,
	                                                      buf(currpos + 3)._1,
	                                                      buf(currpos + 4)._1), j) +
	    theta(1)*f2(buf(currpos + j)._2,buf(currpos + (j-1))._2,Array[String](buf(currpos)._1,
	                                                      buf(currpos + 1)._1,
	                                                      buf(currpos + 2)._1,
	                                                      buf(currpos + 3)._1,
	                                                      buf(currpos + 4)._1), j) +
	    theta(2)*f3(buf(currpos + j)._2,buf(currpos + (j-1))._2,Array[String](buf(currpos)._1,
	                                                      buf(currpos + 1)._1,
	                                                      buf(currpos + 2)._1,
	                                                      buf(currpos + 3)._1,
	                                                      buf(currpos + 4)._1), j) +
	    theta(3)*f4(buf(currpos + j)._2,buf(currpos + (j-1))._2,Array[String](buf(currpos)._1,
	                                                      buf(currpos + 1)._1,
	                                                      buf(currpos + 2)._1,
	                                                      buf(currpos + 3)._1,
	                                                      buf(currpos + 4)._1), j) +
	    theta(4)*f5(buf(currpos + j)._2,buf(currpos + (j-1))._2,Array[String](buf(currpos)._1,
	                                                      buf(currpos + 1)._1,
	                                                      buf(currpos + 2)._1,
	                                                      buf(currpos + 3)._1,
	                                                      buf(currpos + 4)._1), j) +
	    theta(5)*f6(buf(currpos + j)._2,buf(currpos + (j-1))._2,Array[String](buf(currpos)._1,
	                                                      buf(currpos + 1)._1,
	                                                      buf(currpos + 2)._1,
	                                                      buf(currpos + 3)._1,
	                                                      buf(currpos + 4)._1), j) +
	    theta(6)*f7(buf(currpos + j)._2,buf(currpos + (j-1))._2,Array[String](buf(currpos)._1,
	                                                      buf(currpos + 1)._1,
	                                                      buf(currpos + 2)._1,
	                                                      buf(currpos + 3)._1,
	                                                      buf(currpos + 4)._1), j)    

 	}
 	var Z = normalize(Array[String](buf(currpos)._1,
	                            buf(currpos + 1)._1,
	                            buf(currpos + 2)._1,
	                            buf(currpos + 3)._1,
	                            buf(currpos + 4)._1),theta)
 	/*var regpenalty = 0.0
    for(t <- 0 until m) {
        regpenalty += theta(t)*theta(t)/20
    }  - regpenalty*/ 
    return -1.0 * (features.reduceLeft(_+_) - log(Z))
} 

def marg(y: String, yprime: String, theta: DenseVector[Double]): Double = {
//proba yprimt occurs in theata at each place * probability of label is in the distributions?
val templist : List[label] = List(y, yprime)
var result = for(y1<-templist;y2<-templist;y3<-templist;y4<-templist;y5<-templist) yield {
				probability(Array[String](y1,y2,y3,y4,y5),Array[String](buf(currpos)._1,
		                                       buf(currpos + 1)._1,
		                                       buf(currpos + 2)._1,
		                                       buf(currpos + 3)._1,
		                                       buf(currpos + 4)._1),theta)
			}
var toreturn = result.reduceLeft(_+_)
println("marg" + toreturn)
return toreturn
}

def log_likelihood_gradient(theta: DenseVector[Double]):DenseVector[Double] =  {
    println("gradient")
    println("BUFF LEN: " + buf.length)
    println(buf(20)._1)
    if (currpos == buf.length) { break }
	var first = for(j <- 0 until k) yield {
	   	(f1(buf(currpos + j)._2,buf(currpos + (j-1))._2,Array[String](buf(currpos)._1,
	                                                     	buf(currpos + 1)._1,
	                                                     	buf(currpos + 2)._1,
	                                                     	buf(currpos + 3)._1,
	                                                     	buf(currpos + 4)._1), j),
	   	f2(buf(currpos + j)._2,buf(currpos + (j-1))._2,Array[String](buf(currpos)._1,
	                                                     	buf(currpos + 1)._1,
	                                                     	buf(currpos + 2)._1,
	                                                     	buf(currpos + 3)._1,
	                                                     	buf(currpos + 4)._1), j),
	   	f3(buf(currpos + j)._2,buf(currpos + (j-1))._2,Array[String](buf(currpos)._1,
	                                                     	buf(currpos + 1)._1,
	                                                     	buf(currpos + 2)._1,
	                                                     	buf(currpos + 3)._1,
	                                                     	buf(currpos + 4)._1), j),
	   	f4(buf(currpos + j)._2,buf(currpos + (j-1))._2,Array[String](buf(currpos)._1,
	                                                     	buf(currpos + 1)._1,
	                                                     	buf(currpos + 2)._1,
	                                                     	buf(currpos + 3)._1,
	                                                     	buf(currpos + 4)._1), j),
	   	f5(buf(currpos + j)._2,buf(currpos + (j-1))._2,Array[String](buf(currpos)._1,
	                                                     	buf(currpos + 1)._1,
	                                                     	buf(currpos + 2)._1,
	                                                     	buf(currpos + 3)._1,
	                                                     	buf(currpos + 4)._1), j),
	   	f6(buf(currpos + j)._2,buf(currpos + (j-1))._2,Array[String](buf(currpos)._1,
	                                                     	buf(currpos + 1)._1,
	                                                     	buf(currpos + 2)._1,
	                                                     	buf(currpos + 3)._1,
	                                                     	buf(currpos + 4)._1), j),
	   	f7(buf(currpos + j)._2,buf(currpos + (j-1))._2,Array[String](buf(currpos)._1,
	                                                     	buf(currpos + 1)._1,
	                                                     	buf(currpos + 2)._1,
	                                                     	buf(currpos + 3)._1,
	                                                     	buf(currpos + 4)._1), j))
	}

	var second: ListBuffer[(Double,Double,Double,Double,Double,Double,Double)]  = ListBuffer()
	for(j <- 0 until k) {
		var feat = for(y1<-labels;y2<-labels) yield {
			(f1(y1,y2,Array[String](buf(currpos)._1,
		                                       buf(currpos + 1)._1,
		                                       buf(currpos + 2)._1,
		                                       buf(currpos + 3)._1,
		                                       buf(currpos + 4)._1), j)*marg(y1,y2,theta),
																					
			f2(y1,y2,Array[String](buf(currpos)._1,
		                                       buf(currpos + 1)._1,
		                                       buf(currpos + 2)._1,
		                                       buf(currpos + 3)._1,
		                                       buf(currpos + 4)._1), j)*marg(y1,y2,theta),
			f3(y1,y2,Array[String](buf(currpos)._1,
		                                       buf(currpos + 1)._1,
		                                       buf(currpos + 2)._1,
		                                       buf(currpos + 3)._1,
		                                       buf(currpos + 4)._1), j)*marg(y1,y2,theta),
			f4(y1,y2,Array[String](buf(currpos)._1,
		                                       buf(currpos + 1)._1,
		                                       buf(currpos + 2)._1,
		                                       buf(currpos + 3)._1,
		                                       buf(currpos + 4)._1), j)*marg(y1,y2,theta),
			f5(y1,y2,Array[String](buf(currpos)._1,
		                                       buf(currpos + 1)._1,
		                                       buf(currpos + 2)._1,
		                                       buf(currpos + 3)._1,
		                                       buf(currpos + 4)._1), j)*marg(y1,y2,theta),
			f6(y1,y2,Array[String](buf(currpos)._1,
		                                       buf(currpos + 1)._1,
		                                       buf(currpos + 2)._1,
		                                       buf(currpos + 3)._1,
		                                       buf(currpos + 4)._1), j)*marg(y1,y2,theta),
			f7(y1,y2,Array[String](buf(currpos)._1,
		                                       buf(currpos + 1)._1,
		                                       buf(currpos + 2)._1,
		                                       buf(currpos + 3)._1,
		                                       buf(currpos + 4)._1), j)*marg(y1,y2,theta))
		}
		println("HERE!" + (feat.map(_._1).sum,
					feat.map(_._2).sum,
					feat.map(_._3).sum,
					feat.map(_._4).sum,
					feat.map(_._5).sum,
					feat.map(_._6).sum,
					feat.map(_._7).sum))
		second += ((feat.map(_._1).sum,
					feat.map(_._2).sum,
					feat.map(_._3).sum,
					feat.map(_._4).sum,
					feat.map(_._5).sum,
					feat.map(_._6).sum,
					feat.map(_._7).sum))

	}
	   		   
    currpos += 1 
    var toreturn = DenseVector[Double](first.map(_._1).sum - second.map(_._1).sum, first.map(_._2).sum - second.map(_._2).sum,
									   first.map(_._3).sum - second.map(_._3).sum, first.map(_._4).sum - second.map(_._4).sum,																			
									   first.map(_._5).sum - second.map(_._5).sum, first.map(_._6).sum - second.map(_._6).sum,
									   first.map(_._7).sum - second.map(_._7).sum)																					  
    println("currpos" + currpos)
    println("new theta " + toreturn)
    second.clear()
    return toreturn:*=(-1.0)
}


 def train() {
 	val objectivef = new DiffFunction[DenseVector[Double]] {
             def calculate(x: DenseVector[Double]) = {
                 //TODO negate log log_likelihood to minimize
                 (neglog_likelihood(x),log_likelihood_gradient(x));
             }
    }
    val lbfgs = new LBFGS[DenseVector[Double]](maxIter=(-1), m=7)
    var weights = lbfgs.minimize(objectivef,DenseVector(1,1,1,1,1,1,1))
    println(weights)
}
}