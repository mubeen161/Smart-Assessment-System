//Question bank
var questionBank= [
    {
        question : 'What is the half quater of  8000',
        option : ['2000','1000','500','800'],
        answer : '1000'
    },
    {
        question : 'What is the Answer   1+3=4   2+4=10   3+5=18   4+6=?',
        option : ['32','20','46','28'],
        answer : '28'
    },
    {
        question : 'How many 5s appear between 1-100',
        option : ['21','10','11','20'],
        answer : '20'
    },
    {
        question : 'Hitler party which came into power in 1933 is known as',
        option : ['Labour Party','Nazi Party','Ku-Klux-Klan','Democratic Party'],
        answer : 'Nazi Party'
    },
    {
        question : 'What is on the WWF (world wildlife fund) Logo',
        option : ['Koala','Tiger','Panda','Lion'],
        answer : 'Panda'
    },
    {
        question : 'The most victories in the world cup have been won by',
        option : ['Brazil','argentina','Saudi','italy'],
        answer : 'Brazil'
    },
    {
        question : 'What is spider-man Real Name',
        option : ['Tony stark','Bruce wayne','Thor','Peter Parker'],
        answer : 'Peter parker'
    },
    {
        question : 'What number comes next in the Series 1 -> 1 -> 2 -> 3 -> 5 -> 8 -> 13 -> ?',
        option : ['18','13','21','22'],
        answer : '21'
    },
    {
        question : 'Which state in India has a "Separate Constitution',
        option : ['Uttarpradesh','Jammu and Kashmir','Haryana','Rajasthan'],
        answer : 'Jammu and Kashmir'
    },
    {
        question : 'Which Animal dies after eating Chocalete',
        option : ['kangaroo','Cat','Dog','Camel'],
        answer : 'Dog'
    },
]   

var question= document.getElementById('question');
var quizContainer= document.getElementById('quiz-container');
var scorecard= document.getElementById('scorecard');
var option0= document.getElementById('option0');
var option1= document.getElementById('option1');
var option2= document.getElementById('option2');
var option3= document.getElementById('option3');
var next= document.querySelector('.next');
var points= document.getElementById('score');
var span= document.querySelectorAll('span');
var i=0;
var score= 0;

//function to display questions
function displayQuestion(){
    for(var a=0;a<span.length;a++){
        span[a].style.background='none';
    }
    question.innerHTML= 'Q.'+(i+1)+' '+questionBank[i].question;
    option0.innerHTML= questionBank[i].option[0];
    option1.innerHTML= questionBank[i].option[1];
    option2.innerHTML= questionBank[i].option[2];
    option3.innerHTML= questionBank[i].option[3];
    stat.innerHTML= "Question"+' '+(i+1)+' '+'of'+' '+questionBank.length;
}

//function to calculate scores
function calcScore(e){
    if(e.innerHTML===questionBank[i].answer && score<questionBank.length)
    {
        score= score+1;
        document.getElementById(e.id).style.background= 'limegreen';
    }
    else{
        document.getElementById(e.id).style.background= 'tomato';
    }
    setTimeout(nextQuestion,300);
}

//function to display next question
function nextQuestion(){
    if(i<questionBank.length-1)
    {
        i=i+1;
        displayQuestion();
    }
    else{
        points.innerHTML= score+ '/'+ questionBank.length;
        quizContainer.style.display= 'none';
        scoreboard.style.display= 'block'
    }
}

//click events to next button
next.addEventListener('click',nextQuestion);

//Back to Quiz button event
function backToQuiz(){
    location.reload();
}

//function to check Answers
function checkAnswer(){
    var answerBank= document.getElementById('answerBank');
    var answers= document.getElementById('answers');
    answerBank.style.display= 'block';
    scoreboard.style.display= 'none';
    for(var a=0;a<questionBank.length;a++)
    {
        var list= document.createElement('li');
        list.innerHTML= questionBank[a].answer;
        answers.appendChild(list);
    }
}


displayQuestion();