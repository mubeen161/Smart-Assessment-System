<html>
	<head>
		<title>Happiness Index</title>
		<style type="text/css">
			body {
			background-color: #DBF9FC;
				
			}
			
			h1 {font-size:2.9em;text-align: center;font: optional;color: #33475b;}
			h2 {font-size:1.8em;text-align: center;color:#33475b;}
			label{
				width:100px;
				display:inline-block;
			}
			input[type=text]{
				width: 500px;
				padding:5px 3px;
				margin:10px 120px;
				box-sizing:border-box;
				border:5px solid #555;
				outline:none;
				font-size:22px;
				position:relative; 
				
			}
			input[type=text1]{
				width: 500px;
				padding:5px 3px;
				margin:10px 500px;
				box-sizing:border-box;
				border:5px solid #555;
				outline:none;
				font-size:22px;
				position:relative; 
				
			}
			input[type=text]:focus{
				background-color:wheat;
				position:relative; 
			}
			input[type=submit]{
				width: 150;
				padding:auto;
				margin:8px 10px;
				box-sizing:border-box;
				border:5px solid #555;
  				outline:none;  
				background-color:cyan;
				color:brown;
				font-size:25px;
				position:relative; left:640px; 
				
			}
			input[type=submit]:focus{

			}
			.center {
			display: block;
			margin-left: 475px;
			margin-right: 580px;
			}
			.center1 {
			display: block;
			margin-left: 400px;
			margin-right: 500px;
			}
			
			
		</style>
</head>
<body>
<form action="" method="post">
	<h1><b>Happiness Index</b></h1>
	<h2><b>Rate Yourself(out of 10):</b></h2>
<input type="text" placeholder="Social Support" name="no1" >
<input type="text" placeholder="Sleep" name="no2" ><br>
<input type="text" placeholder="Freedom of Choice" name="no3">
<input type="text" placeholder="Physical Well-Being" name="no4"><br>
<input type="text" placeholder="Personal Safety" name="no5">
<input type="text" placeholder="Generosity" name="no6"><br>
<input type="text" placeholder="Healthy Life" name="no7">
<input type="text" placeholder="Unexplain Happiness" name="no8"><br>
<input type="text" Placeholder="Smile Count"name="no9">
<input type="text" placeholder="Friendly" name="no10"><br>
<input type="submit" name="submit" value="SUBMIT ">
</form>
<img src="happiness-report.jpg" alt="Happiness Index of India 2022" width="500px" height="800px" class="center">
<img src="rating.jpg" alt="Happiness of person" width="650px" height="300px" class="center1">
</body>
</html>
<?php
$sum="";
if(isset($_POST['submit']))
{
	$no1=$_POST['no1'];
	$no2=$_POST['no2'];
	$no3=$_POST['no3'];
	$no4=$_POST['no4'];
	$no5=$_POST['no5'];
	$no6=$_POST['no6'];
	$no7=$_POST['no7'];
	$no8=$_POST['no8'];
	$no9=$_POST['no9'];
	$no10=$_POST['no10'];
	$sum=(($no1+$no2+$no3+$no4+$no5+$no6+$no7+$no8+$no9+$no10)/10);
	
}
?>
<h2>Result:</h2>
<input type="text1" value="<?php echo $sum; ?>">	