
function showDiv(elem)
{
	if(elem.value=="S")
	{	
		document.getElementById('hiddev_super').style.display = "block";
		document.getElementById('hiddev_target').style.display = "block";
		document.getElementById('hiddev_unsuper').style.display = "none";
		document.getElementById('hiddev_feature').style.display = "none";
	}
	else if(elem.value=="U")
	{	
		document.getElementById('hiddev_super').style.display = "none";
		document.getElementById('hiddev_target').style.display = "none";
		document.getElementById('hiddev_feature').style.display = "none";
		document.getElementById('hiddev_unsuper').style.display = "block";
	}
	else if(elem.value=="F")
	{
		document.getElementById('hiddev_super').style.display = "none";
		document.getElementById('hiddev_unsuper').style.display = "none";		
		document.getElementById('hiddev_target').style.display = "block";
		document.getElementById('hiddev_feature').style.display = "block";
	}
	else
	{	
		document.getElementById('hiddev_super').style.display = "none";
		document.getElementById('hiddev_target').style.display = "none";
		document.getElementById('hiddev_feature').style.display = "none";
		document.getElementById('hiddev_unsuper').style.display = "none";
	}	
}

function showSuperDiv(elem)
{
	if(elem.value=='C')
	{
		document.getElementById('hiddev_class').style.display="block";
		document.getElementById('hiddev_reg').style.display="none";
	}
	else if(elem.value=='R')
	{
		document.getElementById('hiddev_class').style.display="none";
		document.getElementById('hiddev_reg').style.display="block";
	}
}

function showUnsuperDiv(elem)
{
	if(elem.value=='C')
	{
		document.getElementById('hiddev_clust').style.display = "block";
		document.getElementById('hiddev_features').style.display = "none";
	}
	else if(elem.value=='F')
	{
		document.getElementById('hiddev_clust').style.display = "none";
		document.getElementById('hiddev_features').style.display = "block";
	}
	
}
