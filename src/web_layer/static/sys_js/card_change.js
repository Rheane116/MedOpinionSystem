// 切换报告部分
document.getElementById('hotspotBtn').addEventListener('click', function() {
  document.getElementById('hotspotCard').style.display = 'flex';
  document.getElementById('sentimentCard').style.display = 'none';
  this.classList.remove('btn-outline-primary');
  this.classList.add('btn-primary', 'active');
  document.getElementById('sentimentBtn').classList.remove('btn-primary', 'active');
  document.getElementById('sentimentBtn').classList.add('btn-outline-primary');
});

document.getElementById('sentimentBtn').addEventListener('click', function() {
  document.getElementById('hotspotCard').style.display = 'none';
  document.getElementById('sentimentCard').style.display = 'flex';
  this.classList.remove('btn-outline-primary');
  this.classList.add('btn-primary', 'active');
  document.getElementById('hotspotBtn').classList.remove('btn-primary', 'active');
  document.getElementById('hotspotBtn').classList.add('btn-outline-primary');
});

// 初始化显示热点识别部分
document.getElementById('hotspotCard').style.display = 'flex';