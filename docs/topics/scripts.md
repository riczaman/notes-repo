```
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>Applied AI/ML Hub - Banner</title>
  <style>
    /* Safe, self-contained styles. Works inside Confluence HTML macro. */
    .ai-banner-link{ text-decoration: none; display: block; }
    .ai-banner{
      box-sizing: border-box;
      width: 100%;
      max-width: 1200px;
      margin: 12px auto;
      padding: 18px 24px;
      border-radius: 10px;
      background: linear-gradient(180deg,#012048 0%, #002d6a 100%); /* navy-blue gradient */
      color: #ffffff;
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif;
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 16px;
      box-shadow: 0 6px 18px rgba(0,0,0,0.18);
      border: 1px solid rgba(255,255,255,0.04);
    }
    .ai-banner__left{
      display:flex;
      align-items: center;
      gap: 16px;
    }
    .ai-banner__logo{
      flex: 0 0 auto;
      width: 56px;
      height: 56px;
      border-radius: 8px;
      background: rgba(255,255,255,0.06);
      display:flex;
      align-items:center;
      justify-content:center;
      font-weight:700;
      font-size:18px;
      letter-spacing:0.6px;
    }
    .ai-banner__title{
      font-size:20px;
      line-height:1.05;
      margin:0;
      padding:0;
      font-weight:700;
    }
    .ai-banner__subtitle{
      margin:0;
      font-size:12px;
      opacity:0.9;
      font-weight:500;
    }
    .ai-banner__cta{
      flex:0 0 auto;
      background: rgba(255,255,255,0.06);
      padding:10px 16px;
      border-radius:8px;
      font-weight:600;
      font-size:13px;
      border: 1px solid rgba(255,255,255,0.06);
      transition: transform 120ms ease, box-shadow 120ms ease;
    }
    .ai-banner__cta:active{ transform: translateY(1px); }
    .ai-banner-link:focus .ai-banner, .ai-banner-link:hover .ai-banner{ box-shadow: 0 10px 28px rgba(0,0,0,0.25); transform: translateY(-2px); }

    /* Responsive tweaks */
    @media (max-width:720px){
      .ai-banner{ flex-direction: column; align-items:flex-start; padding:14px; }
      .ai-banner__left{ gap:12px; }
      .ai-banner__title{ font-size:18px; }
      .ai-banner__logo{ width:48px; height:48px; font-size:16px; }
      .ai-banner__cta{ width:100%; text-align:center; }
    }
  </style>
</head>
<body>
  <!-- Replace the href value with the URL you want the banner to open -->
  <a class="ai-banner-link" href="YOUR_URL_HERE" target="_blank" rel="noopener noreferrer">
    <div class="ai-banner" role="banner" aria-label="Applied AI/ML Hub - AI Sandbox">
      <div class="ai-banner__left">
        <div class="ai-banner__logo" aria-hidden="true">AI</div>
        <div>
          <h1 class="ai-banner__title">Applied AI/ML Hub</h1>
          <p class="ai-banner__subtitle">AI Sandbox</p>
        </div>
      </div>
      <div class="ai-banner__cta" role="button" aria-hidden="true">Open Sandbox</div>
    </div>
  </a>

  <!-- Minimal script-free file. Confluence may strip <script> tags; this banner is pure HTML/CSS. -->
</body>
</html>
```