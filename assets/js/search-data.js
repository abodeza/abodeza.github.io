// get the ninja-keys element
const ninja = document.querySelector('ninja-keys');

// add the home and posts menu items
ninja.data = [{
    id: "nav-about",
    title: "about",
    section: "Navigation",
    handler: () => {
      window.location.href = "/";
    },
  },{id: "nav-blog",
          title: "blog",
          description: "",
          section: "Navigation",
          handler: () => {
            window.location.href = "/blog/";
          },
        },{id: "nav-projects",
          title: "projects",
          description: "A growing collection of my projects",
          section: "Navigation",
          handler: () => {
            window.location.href = "/projects/";
          },
        },{id: "nav-repositories",
          title: "repositories",
          description: "Here you will find a listing of my most recent work on GitHub",
          section: "Navigation",
          handler: () => {
            window.location.href = "/repositories/";
          },
        },{id: "nav-cv",
          title: "cv",
          description: "My CV rendered from standard json. A PDF copy is also available by clicking on the PDF icon to the right.",
          section: "Navigation",
          handler: () => {
            window.location.href = "/cv/";
          },
        },{id: "nav-awards-amp-certificates",
          title: "awards &amp; certificates",
          description: "Awards and certificates received.",
          section: "Navigation",
          handler: () => {
            window.location.href = "/certificates/";
          },
        },{id: "post-masking-diseases-on-plant-leaves",
        
          title: 'Masking Diseases on Plant Leaves <svg width="1.2rem" height="1.2rem" top=".5rem" viewBox="0 0 40 40" xmlns="http://www.w3.org/2000/svg"><path d="M17 13.5v6H5v-12h6m3-3h6v6m0-6-9 9" class="icon_svg-stroke" stroke="#999" stroke-width="1.5" fill="none" fill-rule="evenodd" stroke-linecap="round" stroke-linejoin="round"></path></svg>',
        
        description: "",
        section: "Posts",
        handler: () => {
          
            window.open("https://medium.com/@abodeza/masking-diseases-on-plant-leaves-6b43b7d8212f?source=rss-d053c50bd307------2", "_blank");
          
        },
      },{id: "books-the-godfather",
          title: 'The Godfather',
          description: "",
          section: "Books",handler: () => {
              window.location.href = "/books/the_godfather/";
            },},{id: "news-i-graduated-from-rpi-with-a-b-sc-in-ee-sparkles-smile",
          title: 'I graduated from RPI with a B.Sc. in EE! :sparkles: :smile:',
          description: "",
          section: "News",},{id: "news-i-finished-my-training-at-tuwaiq-academy",
          title: 'I finished my training at Tuwaiq Academy',
          description: "",
          section: "News",handler: () => {
              window.location.href = "/news/announcement_2/";
            },},{id: "projects-image-captioning-vit-wip",
          title: 'Image Captioning (ViT) [WIP]',
          description: "Walkthrough",
          section: "Projects",handler: () => {
              window.location.href = "/projects/ViT/";
            },},{id: "projects-a-bi-objective-optimization-approach-for-enhancing-fedul-wip",
          title: 'A Bi-Objective Optimization Approach for Enhancing FedUL [WIP]',
          description: "Algorithm Analysis",
          section: "Projects",handler: () => {
              window.location.href = "/projects/fedul/";
            },},{id: "projects-gharsa-39-s-eye",
          title: 'Gharsa&amp;#39;s Eye',
          description: "Project Walkthrough",
          section: "Projects",handler: () => {
              window.location.href = "/projects/gharsa/";
            },},{id: "projects-recommender-system-wip",
          title: 'Recommender System [WIP]',
          description: "Code",
          section: "Projects",handler: () => {
              window.location.href = "/projects/rec/";
            },},{id: "projects-transfer-learning-amp-auto-ml-using-optuna-wip",
          title: 'Transfer Learning &amp;amp; Auto-ML Using Optuna [WIP]',
          description: "Walkthrough",
          section: "Projects",handler: () => {
              window.location.href = "/projects/tl_adv_tech/";
            },},{id: "projects-quantum-inspired-machine-learning-using-tensor-networks",
          title: 'Quantum-Inspired Machine Learning Using Tensor Networks',
          description: "Walkthrough",
          section: "Projects",handler: () => {
              window.location.href = "/projects/tn_qc/";
            },},{id: "projects-predictive-refrigerant-leak-modelling-in-vrf-systems",
          title: 'Predictive Refrigerant Leak Modelling in VRF Systems',
          description: "A Project Overview",
          section: "Projects",handler: () => {
              window.location.href = "/projects/vrf/";
            },},{
        id: 'social-discord',
        title: 'Discord',
        section: 'Socials',
        handler: () => {
          window.open("https://discord.com/users/acehun1", "_blank");
        },
      },{
        id: 'social-email',
        title: 'email',
        section: 'Socials',
        handler: () => {
          window.open("mailto:%61%62%64%75%6C%6C%61%68.%61%6C%7A%61%68%72%61%6E%69.%70@%67%6D%61%69%6C.%63%6F%6D", "_blank");
        },
      },{
        id: 'social-github',
        title: 'GitHub',
        section: 'Socials',
        handler: () => {
          window.open("https://github.com/abodeza", "_blank");
        },
      },{
        id: 'social-linkedin',
        title: 'LinkedIn',
        section: 'Socials',
        handler: () => {
          window.open("https://www.linkedin.com/in/a-a-alzahrani", "_blank");
        },
      },{
        id: 'social-medium',
        title: 'Medium',
        section: 'Socials',
        handler: () => {
          window.open("https://medium.com/@abodeza", "_blank");
        },
      },{
      id: 'light-theme',
      title: 'Change theme to light',
      description: 'Change the theme of the site to Light',
      section: 'Theme',
      handler: () => {
        setThemeSetting("light");
      },
    },
    {
      id: 'dark-theme',
      title: 'Change theme to dark',
      description: 'Change the theme of the site to Dark',
      section: 'Theme',
      handler: () => {
        setThemeSetting("dark");
      },
    },
    {
      id: 'system-theme',
      title: 'Use system default theme',
      description: 'Change the theme of the site to System Default',
      section: 'Theme',
      handler: () => {
        setThemeSetting("system");
      },
    },];
