<template>
  <div>
    <nav class="navbar is-fixed-top is-dark is-bold">
        <div class="navbar-brand">
            <a class="navbar-item" href="">
                <span class="logo">
                    Le Bl<i class="fas fa-paw"></i>g
                </span>
            </a>
            <div class="navbar-burger burger" data-target="navbarLanguage">
                <span></span>
                <span></span>
                <span></span>
            </div>
        </div>
        <div id="navbarLanguage" class="navbar-menu">
            <div class="navbar-start">
                <div class="navbar-item has-dropdown is-hoverable">
                    <a class="navbar-link">
                        Ngôn Ngữ
                    </a>
                    <div class="navbar-dropdown is-boxed">
                        <a class="navbar-item">
                            <router-link to="/"><span>English</span></router-link>
                        </a>
                        <a class="navbar-item">
                            <router-link to="/vi"><span>Tiếng Việt</span></router-link>
                        </a>
                    </div>
                </div>
            </div>
            <div class="navbar-end">
                <div class="navbar-item">
                    <div class="field is-grouped">
                        <p class="control">
                            <a class="bd-tw-button button" href="https://github.com/quinle/">
                                <span class="icon"><i class="fab fa-github"></i></span>
                            </a>
                        </p>
                        <p class="control">
                            <a class="bd-tw-button button" href="https://www.linkedin.com/in/nguyet-quinn-le-74ab54168/">
                                <span class="icon"><i class="fab fa-linkedin"></i></span>
                            </a>
                        </p>
                        <p class="control">
                            <a class="bd-tw-button button" href="https://www.facebook.com/quinn.le.95/">
                                <span class="icon"><i class="fab fa-facebook"></i></span>
                            </a>
                        </p>
                    </div>
                </div>
            </div>
        </div>
    </nav>
    <section class="hero is-light is-medium">
        <div class="hero-body">
            <div class="container">
                <span class="randomimage"><i class="fas fa-mug-hot" ></i></span>
                <span class="welcome">Let's have coffee sometime</span>
                <!--<span class="randomimage">Frew<i class="fas fa-angle-double-up" ></i>rd</span>
                <span class="welcome">What is your reward function?</span>-->
            </div>
        </div>
    </section>
    <section class="section">
      <div class="container">
        <div class="columns">
          <div class="column is-2 side-nav">
            <side-menu></side-menu>
          </div>
          <div class="column is-10 markdown" :key="latex">
            <router-view></router-view>
          </div>
        </div>
      </div>
    </section>

    <footer class="footer">
      <div class="container">
        <div class="content has-text-centered">
          <p>
            <strong>Le Bl<i class="fas fa-paw"></i>g</strong> by <a href="https://www.facebook.com/quinn.le.95/" target="_blank">Quinn Le</a>. Mã nguồn bản quyền
            <a href="http://opensource.org/licenses/mit-license.php">MIT</a>.
          </p>
          <p>
            <a class="icon" href="https://github.com/quinle/">
              <i class="fab fa-github"></i>
            </a>
          </p>
        </div>
      </div>
    </footer>
  </div>
</template>

<script>
import SideMenu from './ViSideMenu'
export default {
  components: { SideMenu },
  methods: {
    reRender () {
      if (window.MathJax) {
        window.MathJax.Hub.Queue(['Typeset', window.MathJax.Hub], () => console.log('done'))
      }
    }
  },
  mounted () {
    this.reRender()
  },
  watch: {
    latex: function () {
      this.$nextTick().then(() => {
        this.reRender()
      })
    }
  }
}
</script>
<!-- Add "scoped" attribute to limit CSS to this component only -->
<style scoped>
  .logo
  {
  font-size: 30px;
  color: #FDFEFE;
  }
  .randomimage
  {
  display:table;
  margin:0 auto;
  font-size: 200px;
  color: #000000;
  }

  .welcome
  {
  display:table;
  margin:0 auto;
  font-size: 62.5px;
  font-weight: bold;
  color: #695e5e;
  }
</style>
